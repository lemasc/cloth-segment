"""Compare two finished runs from their files only: Markdown tables and training plots.

Reads `meta.json`, `log.jsonl` and `eval/<version>/[val/]<dataset>/metrics.json`
of each run. Never writes into a run folder and never estimates a number:
anything missing from the files is shown as missing.

- Test metrics: per run, the newest eval version that has a test
  `metrics.json` for every dataset (eval-v1 and eval-v2 score a checkpoint the
  same way, M3).
- Val per dataset at the best step: from `log.jsonl` (per-dataset entries,
  eval-v2 training), otherwise from `eval/<version>/val/*/metrics.json` whose
  `checkpoint_step` is that step.
- `miou_without`: mIoU over the classes in `in_mean`, minus the excluded ones.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from cloth_segment.data.dataset import DATASET_NAMES

META_FIELDS: tuple[str, ...] = (
    "config_hash",
    "commit",
    "data_version",
    "steps",
    "best_step",
    "params",
    "trainable_params",
    "train_seconds",
    "eval_seconds",
    "best_val_select",
)
COLOURS = {"lip": "#2a78d6", "cihp": "#eb6834", "dfmm": "#1baf7a"}
MARKERS = {"lip": "o", "cihp": "s", "dfmm": "^"}
MEAN_COLOUR, MUTED, GRID, SURFACE, SHADE = "#52514e", "#6b6a64", "#e4e3df", "#fcfcfb", "#eeeeea"
MEAN_SERIES = "mean of datasets"
POOLED_SERIES = "pooled"


# ---------------------------------------------------------------- reading


def read_jsonl(path: Path) -> list[dict]:
    """One dict per non-empty line."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _version_key(name: str) -> int:
    m = re.fullmatch(r"eval-v(\d+)", name)
    return int(m.group(1)) if m else -1


def test_metrics(run_dir: Path, datasets: Sequence[str] = DATASET_NAMES) -> dict[str, dict]:
    """Test `metrics.json` per dataset from the newest eval version that has all of them."""
    eval_dir = run_dir / "eval"
    versions = sorted(
        (p.name for p in eval_dir.iterdir() if _version_key(p.name) >= 0)
        if eval_dir.is_dir()
        else [],
        key=_version_key,
        reverse=True,
    )
    for v in versions:
        files = {d: eval_dir / v / d / "metrics.json" for d in datasets}
        if all(f.exists() for f in files.values()):
            return {d: read_json(f) for d, f in files.items()}
    raise FileNotFoundError(f"{run_dir.name}: no eval version with test metrics for {datasets}")


def _val_files(run_dir: Path, datasets: Sequence[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {d: [] for d in datasets}
    for f in sorted((run_dir / "eval").glob("eval-v*/val/*/metrics.json")):
        if f.parent.name in out:
            out[f.parent.name].append(read_json(f))
    return out


@dataclass(frozen=True)
class Run:
    """Everything a comparison needs from one run folder."""

    name: str
    meta: dict
    log: list[dict]
    test: dict[str, dict]
    val_files: dict[str, list[dict]]


def load_run(run_dir: Path, datasets: Sequence[str] = DATASET_NAMES) -> Run:
    return Run(
        name=run_dir.name,
        meta=read_json(run_dir / "meta.json"),
        log=read_jsonl(run_dir / "log.jsonl"),
        test=test_metrics(run_dir, datasets),
        val_files=_val_files(run_dir, datasets),
    )


# ---------------------------------------------------------------- derived values


def miou_without(metrics: dict, exclude: Sequence[str] = ()) -> float | None:
    """mIoU over `in_mean` without `exclude`; None if no class is left."""
    kept = [metrics["iou"][c] for c in metrics["in_mean"] if c not in exclude]
    return float(np.mean(kept)) if kept else None


def train_entries(log: list[dict]) -> list[dict]:
    return [x for x in log if "loss" in x]


def val_series(log: list[dict]) -> dict[str, dict[int, dict]]:
    """Val log entries by series: each dataset, and `pooled` (entries without a dataset)."""
    out: dict[str, dict[int, dict]] = {}
    for x in log:
        if x.get("split") != "val" or "miou" not in x:
            continue
        out.setdefault(x.get("dataset", POOLED_SERIES), {})[x["step"]] = x
    return out


def select_series(log: list[dict]) -> dict[int, float]:
    """The checkpoint selection value per val step (eval-v2 logs only)."""
    return {x["step"]: x["value"] for x in log if "select" in x}


def val_at_step(run: Run, step: int, datasets: Sequence[str] = DATASET_NAMES) -> dict[str, dict]:
    """Val metrics per dataset at `step`, from the log or else from val metrics.json files."""
    series = val_series(run.log)
    out = {}
    for d in datasets:
        if step in series.get(d, {}):
            out[d] = series[d][step]
            continue
        hits = [m for m in run.val_files.get(d, []) if m.get("checkpoint_step") == step]
        if hits:
            out[d] = hits[-1]
    return out


def loss_blocks(log: list[dict], block: int) -> list[tuple[int, float, int]]:
    """(block start, mean train loss, entries) for each `block`-step window (start, start+block]."""
    tr = train_entries(log)
    if not tr:
        return []
    steps = np.array([x["step"] for x in tr])
    loss = np.array([x["loss"] for x in tr])
    out = []
    for a in range(0, int(steps.max()), block):
        m = (steps > a) & (steps <= a + block)
        if m.any():
            out.append((a, float(loss[m].mean()), int(m.sum())))
    return out


def predicted_as(metrics: dict, cls: str) -> dict:
    """For class `cls`: pixels predicted as it, correct ones, and where its ground truth went."""
    classes = metrics["classes"]
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    i = classes.index(cls)
    gt = int(cm[i].sum())
    share = {c: (float(cm[i, j]) / gt if gt else None) for j, c in enumerate(classes)}
    return {"gt": gt, "predicted": int(cm[:, i].sum()), "correct": int(cm[i, i]), "gt_share": share}


# ---------------------------------------------------------------- Markdown


def _f(v: float | None) -> str:
    return "-" if v is None else f"{v:.4f}"


def _d(a: float | None, b: float | None) -> str:
    return "-" if a is None or b is None else f"{b - a:+.4f}"


def _mean(vals: Sequence[float | None]) -> float | None:
    return None if any(v is None for v in vals) else float(np.mean(vals))


def comparison_markdown(
    old: Run,
    new: Run,
    exclude: Sequence[str] = (),
    block: int = 4000,
    datasets: Sequence[str] = DATASET_NAMES,
) -> str:
    """Tables comparing `old` and `new`; every value is read from the run files."""
    md = [f"# {old.name} vs {new.name} (from files only)\n", "| | old | new |", "|---|---|---|"]
    for k in META_FIELDS:
        md.append(
            f"| {k} | {old.meta.get(k, '(not in meta)')} | {new.meta.get(k, '(not in meta)')} |"
        )
    o0, n0 = old.test[datasets[0]], new.test[datasets[0]]
    for k in ("checkpoint_step", "checkpoint_select", "eval_version"):
        md.append(f"| {k} (test metrics.json) | {o0.get(k)} | {n0.get(k)} |")

    md += ["\n## val mIoU per dataset at best_step\n", "| dataset | old | new | new - old |"]
    md.append("|---|---|---|---|")
    vo = val_at_step(old, old.meta["best_step"], datasets)
    vn = val_at_step(new, new.meta["best_step"], datasets)
    for d in datasets:
        a, b = vo.get(d, {}).get("miou"), vn.get(d, {}).get("miou")
        md.append(f"| {d} | {_f(a)} | {_f(b)} | {_d(a, b)} |")
    ma = _mean([vo.get(d, {}).get("miou") for d in datasets])
    mb = _mean([vn.get(d, {}).get("miou") for d in datasets])
    md.append(f"| mean of datasets | {_f(ma)} | {_f(mb)} | {_d(ma, mb)} |")

    md += ["\n## test mIoU / mean Dice / pixel accuracy\n"]
    md.append(
        "| dataset | images | old mIoU | new mIoU | Δ | old mDice | new mDice | Δ "
        "| old pixacc | new pixacc |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in datasets:
        o, n = old.test[d], new.test[d]
        md.append(
            f"| {d} | {n['images']} | {_f(o['miou'])} | {_f(n['miou'])} "
            f"| {_d(o['miou'], n['miou'])} "
            f"| {_f(o['mdice'])} | {_f(n['mdice'])} | {_d(o['mdice'], n['mdice'])} "
            f"| {_f(o['pixel_acc'])} | {_f(n['pixel_acc'])} |"
        )

    md += ["\n## test mIoU, mean of datasets\n"]
    if exclude:
        label = ", ".join(exclude)
        md += [f"| | all classes | without {label} |", "|---|---|---|"]
    else:
        md += ["| | all classes |", "|---|---|"]
    rows = {}
    for tag, run in (("old", old), ("new", new)):
        full = _mean([run.test[d]["miou"] for d in datasets])
        part = _mean([miou_without(run.test[d], exclude) for d in datasets]) if exclude else None
        rows[tag] = (full, part)
        md.append(f"| {tag} | {_f(full)} |" + (f" {_f(part)} |" if exclude else ""))
    diff = f"| new - old | {_d(rows['old'][0], rows['new'][0])} |"
    md.append(diff + (f" {_d(rows['old'][1], rows['new'][1])} |" if exclude else ""))

    md += ["\n## test IoU per class\n"]
    md.append("| class | " + " | ".join(f"{d} old | {d} new | Δ" for d in datasets) + " |")
    md.append("|---|" + "---|---|---|" * len(datasets))
    for c in new.test[datasets[0]]["classes"]:
        cells = []
        for d in datasets:
            a, b = old.test[d]["iou"].get(c), new.test[d]["iou"].get(c)
            cells += [_f(a), _f(b), _d(a, b)]
        md.append(f"| {c} | " + " | ".join(cells) + " |")

    for c in exclude:
        md += [f"\n## `{c}` on test\n"]
        md.append(
            "| dataset | gt share of pixels | run | predicted px | correct px | top gt → pred |"
        )
        md.append("|---|---|---|---|---|---|")
        for d in datasets:
            gtp = new.test[d]["gt_pixels"]
            share = gtp[c] / sum(gtp.values()) if sum(gtp.values()) else None
            for tag, run in (("old", old), ("new", new)):
                p = predicted_as(run.test[d], c)
                top = sorted(
                    ((k, v) for k, v in p["gt_share"].items() if v), key=lambda kv: -kv[1]
                )[:3]
                tops = ", ".join(f"{k} {v:.1%}" for k, v in top)
                md.append(
                    f"| {d} | {'-' if share is None else f'{share:.3%}'} | {tag} "
                    f"| {p['predicted']:,} | {p['correct']:,} | {tops} |"
                )

    for tag, run in (("old", old), ("new", new)):
        md += _run_trend(tag, run, exclude, block, datasets)
    return "\n".join(md) + "\n"


def _run_trend(
    tag: str, run: Run, exclude: Sequence[str], block: int, datasets: Sequence[str]
) -> list[str]:
    series = val_series(run.log)
    sel = select_series(run.log)
    names = [d for d in datasets if d in series] or (
        [POOLED_SERIES] if POOLED_SERIES in series else []
    )
    md = [f"\n## {tag} run: val mIoU per check (log.jsonl)\n"]
    if not names:
        return md + ["no val entries in log.jsonl"]
    steps = sorted(set().union(*(series[s].keys() for s in names)))
    lr_at = {x["step"]: x["lr"] for x in train_entries(run.log)}
    head = (
        names + (["select value"] if sel else []) + [f"{c} ({s})" for c in exclude for s in names]
    )
    md += ["| step | " + " | ".join(head) + " | lr |", "|---|" + "---|" * (len(head) + 1)]
    for s in steps:
        cells = [_f(series[n].get(s, {}).get("miou")) for n in names]
        if sel:
            cells.append(_f(sel.get(s)))
        cells += [_f(series[n].get(s, {}).get("iou", {}).get(c)) for c in exclude for n in names]
        lr = lr_at.get(s)
        md.append(f"| {s} | " + " | ".join(cells) + f" | {'-' if lr is None else f'{lr:.2e}'} |")
    md.append("\nchange of val mIoU per check:")
    for n in names:
        pts = {s: x["miou"] for s, x in series[n].items()}
        st = sorted(pts)
        md.append(
            f"- {n}: " + ", ".join(f"{a}→{b} {pts[b] - pts[a]:+.4f}" for a, b in zip(st, st[1:]))
        )
    if sel:
        ss = sorted(sel)
        md.append(
            "- select value: "
            + ", ".join(f"{a}→{b} {sel[b] - sel[a]:+.4f}" for a, b in zip(ss, ss[1:]))
        )
    blocks = loss_blocks(run.log, block)
    if blocks:
        md += [f"\n## {tag} run: mean train loss per {block}-step block\n"]
        md += ["| block | mean loss | entries |", "|---|---|---|"]
        md += [f"| {a}–{a + block} | {v:.4f} | {n} |" for a, v, n in blocks]
    return md


# ---------------------------------------------------------------- plots


def _style(ax, shade: tuple[int, int] | None) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.8)
    if shade:
        ax.axvspan(*shade, color=SHADE, zorder=0)


def _val_lines(run: Run, datasets: Sequence[str]) -> dict[str, dict[int, float]]:
    """Val mIoU lines to plot: per dataset plus the select value, else the pooled mIoU."""
    series = val_series(run.log)
    lines = {d: {s: x["miou"] for s, x in series[d].items()} for d in datasets if d in series}
    sel = select_series(run.log)
    if sel:
        lines[MEAN_SERIES] = sel
    elif POOLED_SERIES in series:
        lines[POOLED_SERIES] = {s: x["miou"] for s, x in series[POOLED_SERIES].items()}
    return lines


def plot_curves(
    run: Run, out: Path, shade_last: int = 8000, datasets: Sequence[str] = DATASET_NAMES
) -> Path:
    """Train loss, val mIoU per check and gain per check of one run, as one PNG."""
    tr = train_entries(run.log)
    ts = np.array([x["step"] for x in tr])
    tl = np.array([x["loss"] for x in tr])
    end = int(run.meta.get("steps") or (ts.max() if len(ts) else 0))
    shade = (max(0, end - shade_last), end) if shade_last and end else None
    fig = Figure(figsize=(16, 5), facecolor=SURFACE)
    axes = fig.subplots(1, 3)

    ax = axes[0]
    _style(ax, shade)
    if len(tl):
        ax.plot(ts, tl, color="#b7d3f6", lw=1, label="every log entry")
        k = min(20, len(tl))
        ma = np.convolve(tl, np.ones(k) / k, mode="valid")
        ax.plot(ts[k - 1 :], ma, color=COLOURS["lip"], lw=2, label=f"{k}-entry moving mean")
        ax.set_ylim(0, max(1.0, float(np.percentile(tl, 99.5))))
        ax.legend(loc="lower left", frameon=False)
    ax.set(title="Train loss", xlabel="step", ylabel="loss")

    lines = _val_lines(run, datasets)
    for ax, gain in ((axes[1], False), (axes[2], True)):
        _style(ax, shade)
        for name, pts in lines.items():
            st = sorted(pts)
            xs, ys = (
                (st[1:], [pts[b] - pts[a] for a, b in zip(st, st[1:])])
                if gain
                else (
                    st,
                    [pts[s] for s in st],
                )
            )
            is_ds = name in COLOURS
            ax.plot(
                xs,
                ys,
                color=COLOURS.get(name, MEAN_COLOUR),
                lw=2,
                ls="-" if is_ds else "--",
                marker=MARKERS.get(name, "D"),
                ms=6,
                label=name.upper() if is_ds else name,
            )
        if lines:
            ax.legend(loc="upper right" if gain else "lower right", frameon=False)
        if gain:
            ax.axhline(0, color=MUTED, lw=1)
            ax.set(title="Gain over the previous check", xlabel="step", ylabel="Δ val mIoU")
        else:
            ax.set(title="Val mIoU per check", xlabel="step", ylabel="val mIoU")
    fig.suptitle(
        f"{run.name}" + (f" (shaded: last {shade_last} steps)" if shade else ""),
        x=0.01,
        ha="left",
        color=MUTED,
    )
    fig.tight_layout()
    path = out / f"curves_{run.name}.png"
    fig.savefig(path, dpi=130)
    return path


def plot_lr(run: Run, out: Path) -> Path:
    """Learning rate over steps of one run, as one PNG."""
    tr = train_entries(run.log)
    fig = Figure(figsize=(6, 2.6), facecolor=SURFACE)
    ax = fig.subplots()
    _style(ax, None)
    ax.plot([x["step"] for x in tr], [x["lr"] for x in tr], color=MEAN_COLOUR, lw=2)
    ax.set(title=f"Learning rate: {run.name}", xlabel="step", ylabel="lr")
    fig.tight_layout()
    path = out / f"lr_{run.name}.png"
    fig.savefig(path, dpi=130)
    return path
