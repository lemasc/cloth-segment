"""Three-model results on the test sets, from saved files only (T9, analysis only).

Inputs (read only):
- `runs/<run>/eval/<version>/<dataset>/metrics.json`: the official scores
  (frozen protocol, `evaluate.py`). These stay the headline numbers.
- `data/<breakdown_dir>/<run>/<dataset>/raw_confusion.json` and
  `per_image_shared.csv`, written by `scripts/raw_class_breakdown.py` (M10).

What this adds, without changing `evaluate.py`:
- **Adjusted scores** (M10 decisions d1, d2): the shared confusion matrix is
  rebuilt from the raw-class counts, leaving some raw classes out, as if their
  pixels were IGNORE. Both the base and the adjusted score come from the
  rebuilt matrix, so they differ only by the dropped pixels (X4's rebuilt
  matrix differs a little from its saved one: it was recomputed on another
  GPU, M10).
- **Paired bootstrap over test images**: resample the images of a test set
  with replacement, add up their per-image counts, and score. The same
  resample is used for every run, so differences between runs are paired.
  This covers only which test images were drawn, not training noise: there is
  one run per model.
- Tables (Markdown) and figures (PNG) for the T9 report.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.figure import Figure

from cloth_segment import compare
from cloth_segment.data.labels import NUM_CLASSES, SHARED_CLASSES
from cloth_segment.evaluate import scores

STATS = ("gt", "pred", "tp")
CONFIG_KEYS = ("runs", "datasets", "breakdown_dir", "exclude", "adjusted", "bootstrap", "out")


# ---------------------------------------------------------------- config and reading


def load_config(path: Path) -> dict:
    """Read a report config and check the keys every step needs."""
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    for alias, r in cfg["runs"].items():
        if not {"name", "model"} <= set(r):
            raise ValueError(f"{path}: run {alias!r} needs 'name' and 'model'")
    for key, a in cfg["adjusted"].items():
        if not {"dataset", "cls", "raw"} <= set(a):
            raise ValueError(f"{path}: adjusted {key!r} needs 'dataset', 'cls' and 'raw'")
        if a["dataset"] not in cfg["datasets"]:
            raise ValueError(f"{path}: adjusted {key!r} dataset is not in datasets")
    return cfg


def read_image_counts(path: Path) -> np.ndarray:
    """`per_image_shared.csv` -> (images, 3, NUM_CLASSES) int64 gt / pred / tp, by image index.

    Classes missing for an image count 0. Fails on an unknown class name or a
    gap in the image indices, so a wrong file cannot be scored.
    """
    df = pd.read_csv(path, dtype={"id": str})
    unknown = set(df["cls"]) - set(SHARED_CLASSES)
    if unknown:
        raise ValueError(f"{path}: unknown classes {sorted(unknown)}")
    n = int(df["index"].max()) + 1
    if df["index"].nunique() != n:
        raise ValueError(f"{path}: image indices are not 0..{n - 1}")
    out = np.zeros((n, len(STATS), NUM_CLASSES), dtype=np.int64)
    col = df["cls"].map({c: i for i, c in enumerate(SHARED_CLASSES)}).to_numpy()
    for s, name in enumerate(STATS):
        np.add.at(out[:, s, :], (df["index"].to_numpy(), col), df[name].to_numpy())
    return out


# ---------------------------------------------------------------- scores


def cm_from_raw(raw_conf: Mapping, drop: Sequence[str] = ()) -> np.ndarray:
    """Shared confusion matrix (rows = ground truth) rebuilt from `raw_confusion.json`.

    Each raw class adds its `where` counts to the row of the shared class it
    maps to. Raw classes in `drop` are left out, as if their pixels were
    IGNORE. Raw classes that map to IGNORE are never counted.
    """
    raw = raw_conf["raw_classes"]
    unknown = set(drop) - set(raw)
    if unknown:
        raise ValueError(f"cannot drop {sorted(unknown)}: not in this raw_confusion")
    idx = {c: i for i, c in enumerate(SHARED_CLASSES)}
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for name, r in raw.items():
        if r["shared"] is None:
            if name in drop:
                raise ValueError(f"raw class {name!r} maps to IGNORE; nothing to drop")
            continue
        if name in drop:
            continue
        for pred, n in r["where"].items():
            cm[idx[r["shared"]], idx[pred]] += n
    return cm


def score_cm(cm: np.ndarray) -> dict:
    """`evaluate.scores` of a numpy confusion matrix (the frozen metric code)."""
    return scores(torch.from_numpy(np.asarray(cm, dtype=np.int64)), SHARED_CLASSES)


def mean_without(metrics: Mapping, key: str, exclude: Sequence[str] = ()) -> float | None:
    """Mean of `metrics[key]` ('iou' or 'dice') over `in_mean` minus `exclude`."""
    kept = [metrics[key][c] for c in metrics["in_mean"] if c not in exclude]
    return float(np.mean(kept)) if kept else None


def miou_from_counts(counts: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """mIoU from summed counts (..., 3, C): same rule as `evaluate.scores`.

    A class is in the mean when it has ground truth and `keep` is True for it.
    Returns NaN where no class qualifies.
    """
    gt, pred, tp = counts[..., 0, :], counts[..., 1, :], counts[..., 2, :]
    union = gt + pred - tp
    iou = np.divide(tp, union, out=np.zeros(union.shape, dtype=np.float64), where=union > 0)
    use = gt > 0 if keep is None else (gt > 0) & keep
    n = use.sum(-1)
    total = np.where(use, iou, 0.0).sum(-1)
    return np.divide(total, n, out=np.full(n.shape, np.nan), where=n > 0)


def keep_mask(exclude: Sequence[str] = ()) -> np.ndarray:
    unknown = set(exclude) - set(SHARED_CLASSES)
    if unknown:
        raise ValueError(f"unknown classes {sorted(unknown)}")
    return np.array([c not in exclude for c in SHARED_CLASSES])


def bootstrap_miou(
    counts: Mapping[str, np.ndarray],
    n: int,
    seed: int,
    exclude: Sequence[str] = (),
    chunk: int = 50,
) -> dict[str, np.ndarray]:
    """(n,) bootstrap mIoU per run for one test set; the same resamples for every run.

    `counts` maps run -> (images, 3, C) per-image counts of the same images in
    the same order. Each resample draws as many images as the test set has,
    with replacement.
    """
    shapes = {k: v.shape for k, v in counts.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"per-image counts differ in shape: {shapes}")
    images = next(iter(counts.values())).shape[0]
    keep = keep_mask(exclude)
    flat = {k: v.reshape(images, -1).astype(np.float64) for k, v in counts.items()}
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n) for k in counts}
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        draws = rng.integers(0, images, size=(m, images))
        w = np.stack([np.bincount(d, minlength=images) for d in draws]).astype(np.float64)
        for k, f in flat.items():
            summed = (w @ f).reshape(m, len(STATS), NUM_CLASSES)
            out[k][start : start + m] = miou_from_counts(summed, keep)
    return out


def interval(samples: np.ndarray, level: float) -> list[float]:
    """Percentile interval of bootstrap samples."""
    a = (1 - level) / 2
    lo, hi = np.percentile(samples, [100 * a, 100 * (1 - a)])
    return [float(lo), float(hi)]


# ---------------------------------------------------------------- building the results


def _scores_entry(m: Mapping, exclude: Sequence[str]) -> dict:
    return {
        "images": m.get("images"),
        "miou": m["miou"],
        "miou_wo": mean_without(m, "iou", exclude),
        "mdice": m["mdice"],
        "mdice_wo": mean_without(m, "dice", exclude),
        "pixel_acc": m["pixel_acc"],
        "iou": m["iou"],
        "dice": m["dice"],
        "in_mean": m["in_mean"],
    }


def build_results(
    cfg: Mapping,
    official: Mapping[str, Mapping[str, Mapping]],
    meta: Mapping[str, Mapping],
    raw_conf: Mapping[str, Mapping[str, Mapping]],
    image_counts: Mapping[str, Mapping[str, np.ndarray]],
) -> dict:
    """Every number of the T9 report. Inputs are keyed by run alias, then dataset.

    `official`: test `metrics.json`; `raw_conf`: `raw_confusion.json`;
    `image_counts`: `read_image_counts` output.
    """
    aliases, datasets = list(cfg["runs"]), list(cfg["datasets"])
    exclude = list(cfg["exclude"])
    bs = cfg["bootstrap"]
    level = float(bs["level"])
    res: dict = {
        "exclude": exclude,
        "datasets": datasets,
        "runs": {},
        "official": {},
        "mean_of_datasets": {},
        "crosscheck": {},
        "bootstrap": {"n": bs["n"], "seed": bs["seed"], "level": level},
        "adjusted": {},
    }
    for a in aliases:
        m0 = official[a][datasets[0]]
        res["runs"][a] = {
            "name": cfg["runs"][a]["name"],
            "model": cfg["runs"][a]["model"],
            "checkpoint_step": m0.get("checkpoint_step"),
            "eval_version": m0.get("eval_version"),
            **{k: meta[a].get(k) for k in compare.META_FIELDS + ("gpu_name",)},
        }
        res["official"][a] = {d: _scores_entry(official[a][d], exclude) for d in datasets}
        per = res["official"][a]
        res["mean_of_datasets"][a] = {
            k: float(np.mean([per[d][k] for d in datasets]))
            for k in ("miou", "miou_wo", "mdice", "mdice_wo", "pixel_acc")
        }

    # Rebuilt matrices and per-image sums must give the official scores (or, for a
    # run recomputed on another GPU, nearly: the difference is reported, not hidden).
    for a in aliases:
        res["crosscheck"][a] = {}
        for d in datasets:
            rebuilt = score_cm(cm_from_raw(raw_conf[a][d]))
            summed = miou_from_counts(image_counts[a][d].sum(0))
            off = official[a][d]
            diffs = [
                abs(rebuilt["iou"][c] - off["iou"][c])
                for c in SHARED_CLASSES
                if off["iou"][c] is not None and rebuilt["iou"][c] is not None
            ]
            res["crosscheck"][a][d] = {
                "max_abs_iou_diff_rebuilt_vs_official": max(diffs) if diffs else 0.0,
                "miou_rebuilt": rebuilt["miou"],
                "miou_per_image_sum": float(summed),
                "same_cm_as_official": bool(
                    np.array_equal(cm_from_raw(raw_conf[a][d]), np.asarray(off["confusion_matrix"]))
                ),
            }

    # Paired bootstrap per test set; the mean over test sets resamples each set apart.
    samples: dict[str, dict[str, np.ndarray]] = {}
    for i, d in enumerate(datasets):
        per_run = bootstrap_miou(
            {a: image_counts[a][d] for a in aliases}, int(bs["n"]), int(bs["seed"]) + i, exclude
        )
        for a in aliases:
            samples.setdefault(a, {})[d] = per_run[a]
    for a in aliases:
        samples[a]["mean"] = np.mean([samples[a][d] for d in datasets], axis=0)
    res["bootstrap"]["miou_wo"] = {
        a: {d: interval(s, level) for d, s in samples[a].items()} for a in aliases
    }
    pairs = [(b, a) for i, a in enumerate(aliases) for b in aliases[i + 1 :]]
    res["bootstrap"]["diff"] = {}
    for b, a in pairs:
        res["bootstrap"]["diff"][f"{b}-{a}"] = {}
        for d in [*datasets, "mean"]:
            diff = samples[b][d] - samples[a][d]
            point = (
                res["mean_of_datasets"][b]["miou_wo"] - res["mean_of_datasets"][a]["miou_wo"]
                if d == "mean"
                else res["official"][b][d]["miou_wo"] - res["official"][a][d]["miou_wo"]
            )
            res["bootstrap"]["diff"][f"{b}-{a}"][d] = {
                "diff": point,
                "interval": interval(diff, level),
                "share_le_0": float((diff <= 0).mean()),
            }

    for key, adj in cfg["adjusted"].items():
        d, cls, drop = adj["dataset"], adj["cls"], list(adj["raw"])
        entry: dict = {"dataset": d, "cls": cls, "raw": drop, "runs": {}}
        for a in aliases:
            rc = raw_conf[a][d]
            wrong = [r for r in drop if rc["raw_classes"][r]["shared"] != cls]
            if wrong:
                raise ValueError(f"adjusted {key}: {wrong} do not map to {cls!r} in {d}")
            base, adjusted = cm_from_raw(rc), cm_from_raw(rc, drop)
            sb, sa = score_cm(base), score_cm(adjusted)
            cidx = SHARED_CLASSES.index(cls)
            entry["raw_px"] = int(base[cidx].sum() - adjusted[cidx].sum())
            entry["share_of_class_gt"] = entry["raw_px"] / int(base[cidx].sum())
            entry["runs"][a] = {
                "iou_base": sb["iou"][cls],
                "iou_adjusted": sa["iou"][cls],
                "dice_base": sb["dice"][cls],
                "dice_adjusted": sa["dice"][cls],
                "miou_wo_base": mean_without(sb, "iou", exclude),
                "miou_wo_adjusted": mean_without(sa, "iou", exclude),
                "recall_of_dropped": {r: rc["raw_classes"][r]["recall"] for r in drop},
            }
        res["adjusted"][key] = entry
    return res


def load_inputs(cfg: Mapping, runs_dir: Path, data_dir: Path) -> tuple[dict, dict, dict, dict]:
    """(official, meta, raw_conf, image_counts) for `build_results`, from the files."""
    official, meta, raw_conf, counts = {}, {}, {}, {}
    datasets = list(cfg["datasets"])
    base = Path(data_dir) / cfg["breakdown_dir"]
    for a, r in cfg["runs"].items():
        run_dir = Path(runs_dir) / r["name"]
        meta[a] = compare.read_json(run_dir / "meta.json")
        official[a] = compare.test_metrics(run_dir, datasets)
        raw_conf[a] = {
            d: compare.read_json(base / r["name"] / d / "raw_confusion.json") for d in datasets
        }
        counts[a] = {
            d: read_image_counts(base / r["name"] / d / "per_image_shared.csv") for d in datasets
        }
        for d in datasets:
            step = raw_conf[a][d]["checkpoint_step"]
            if step != official[a][d]["checkpoint_step"]:
                raise ValueError(f"{a} {d}: breakdown is of step {step}, metrics of another")
            if counts[a][d].shape[0] != official[a][d]["images"]:
                raise ValueError(f"{a} {d}: per-image counts do not cover the test set")
    return official, meta, raw_conf, counts


def write_json(res: Mapping, path: Path) -> None:
    Path(path).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- Markdown


def _f(v: float | None, nd: int = 4) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _ci(iv: Sequence[float]) -> str:
    return f"{iv[0]:.3f}–{iv[1]:.3f}"


def tables_markdown(res: Mapping) -> str:
    """All T9 tables. Every value comes from `res`; nothing is estimated."""
    aliases, datasets = list(res["runs"]), list(res["datasets"])
    wo = ", ".join(res["exclude"])
    lvl = f"{res['bootstrap']['level']:.0%}"
    md = ["# T9 tables (generated by scripts/t9_results.py; do not edit by hand)\n"]

    md += ["## Runs\n", "| | " + " | ".join(aliases) + " |", "|---|" + "---|" * len(aliases)]
    rows = [
        ("model", lambda r: r["model"]),
        ("run", lambda r: f"`{r['name']}`"),
        ("config_hash", lambda r: r["config_hash"]),
        ("data_version", lambda r: r["data_version"]),
        ("eval_version", lambda r: r["eval_version"]),
        ("checkpoint step (best.pt)", lambda r: r["checkpoint_step"]),
        ("params (M)", lambda r: f"{r['params'] / 1e6:.1f}" if r.get("params") else "-"),
        (
            "train time (h)",
            lambda r: f"{r['train_seconds'] / 3600:.2f}" if r.get("train_seconds") else "-",
        ),
        ("GPU", lambda r: r.get("gpu_name") or "-"),
    ]
    for label, get in rows:
        md.append(f"| {label} | " + " | ".join(str(get(res["runs"][a])) for a in aliases) + " |")

    md += [f"\n## Test mIoU without {wo} (headline), with {lvl} bootstrap interval\n"]
    md += [
        "| test set | images | " + " | ".join(aliases) + " |",
        "|---|---|" + "---|" * len(aliases),
    ]
    for d in [*datasets, "mean"]:
        cells = []
        for a in aliases:
            v = (
                res["mean_of_datasets"][a]["miou_wo"]
                if d == "mean"
                else res["official"][a][d]["miou_wo"]
            )
            cells.append(f"{_f(v)} ({_ci(res['bootstrap']['miou_wo'][a][d])})")
        n = "-" if d == "mean" else res["official"][aliases[0]][d]["images"]
        md.append(
            f"| {'mean of test sets' if d == 'mean' else d} | {n} | " + " | ".join(cells) + " |"
        )

    md += ["\n## Test scores, all metrics\n"]
    head = ["mIoU (12)", f"mIoU (no {wo})", "mDice (12)", f"mDice (no {wo})", "pixel acc"]
    keys = ["miou", "miou_wo", "mdice", "mdice_wo", "pixel_acc"]
    md += ["| test set | run | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(head)]
    for d in [*datasets, "mean"]:
        for a in aliases:
            src = res["mean_of_datasets"][a] if d == "mean" else res["official"][a][d]
            md.append(
                f"| {'mean' if d == 'mean' else d} | {a} | "
                + " | ".join(_f(src[k]) for k in keys)
                + " |"
            )

    md += [f"\n## Paired differences in mIoU without {wo} ({lvl} bootstrap interval)\n"]
    md += [
        "| pair | " + " | ".join([*datasets, "mean"]) + " |",
        "|---|" + "---|" * (len(datasets) + 1),
    ]
    for pair, per in res["bootstrap"]["diff"].items():
        cells = [
            f"{per[d]['diff']:+.4f} ({per[d]['interval'][0]:+.3f} to {per[d]['interval'][1]:+.3f})"
            for d in [*datasets, "mean"]
        ]
        md.append(f"| {pair} | " + " | ".join(cells) + " |")

    for metric in ("iou", "dice"):
        md += [f"\n## Test {metric.upper() if metric == 'iou' else 'Dice'} per class\n"]
        md += ["| class | " + " | ".join(f"{d} {a}" for d in datasets for a in aliases) + " |"]
        md.append("|---|" + "---|" * (len(datasets) * len(aliases)))
        for c in SHARED_CLASSES:
            cells = []
            for d in datasets:
                vals = [res["official"][a][d][metric][c] for a in aliases]
                best = max((v for v in vals if v is not None), default=None)
                cells += [f"**{_f(v)}**" if v is not None and v == best else _f(v) for v in vals]
            md.append(f"| {c} | " + " | ".join(cells) + " |")

    md += ["\n## Adjusted scores (M10 d1, d2): dropped raw-class pixels counted as IGNORE\n"]
    md += [
        "| key | test set | class | raw classes dropped | dropped px (share of class) | run "
        f"| IoU base | IoU adjusted | Δ | mIoU no {wo} base | adjusted | Δ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, e in res["adjusted"].items():
        for a in aliases:
            r = e["runs"][a]
            md.append(
                f"| {key} | {e['dataset']} | {e['cls']} | {', '.join(e['raw'])} "
                f"| {e['raw_px']:,} ({e['share_of_class_gt']:.1%}) | {a} "
                f"| {_f(r['iou_base'])} | {_f(r['iou_adjusted'])} "
                f"| {r['iou_adjusted'] - r['iou_base']:+.4f} "
                f"| {_f(r['miou_wo_base'])} | {_f(r['miou_wo_adjusted'])} "
                f"| {r['miou_wo_adjusted'] - r['miou_wo_base']:+.4f} |"
            )

    md += ["\n## Cross-check: rebuilt matrix vs saved metrics.json\n"]
    md += [
        "| run | test set | same matrix | max abs IoU diff | mIoU rebuilt | mIoU per-image sum |"
    ]
    md.append("|---|---|---|---|---|---|")
    for a in aliases:
        for d in datasets:
            c = res["crosscheck"][a][d]
            md.append(
                f"| {a} | {d} | {'yes' if c['same_cm_as_official'] else 'no'} "
                f"| {c['max_abs_iou_diff_rebuilt_vs_official']:.5f} "
                f"| {_f(c['miou_rebuilt'])} | {_f(c['miou_per_image_sum'])} |"
            )
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- figures

# Categorical slots 1-3 of the dataviz reference palette (validated, light mode).
RUN_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED, GRID, SURFACE = "#2b2a27", "#6b6a64", "#e4e3df", "#fcfcfb"
SEQ_CMAP = ("#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _label(res: Mapping, alias: str) -> str:
    return f"{alias} {res['runs'][alias]['model']}"


def _legend(ax, **kw) -> None:
    """Legend of the labelled bars only (error bars carry no label)."""
    ax.legend(frameon=False, labelcolor=INK, **kw)


def plot_headline(res: Mapping, out: Path) -> Path:
    """Grouped bars: mIoU without the excluded classes per test set and run, with intervals."""
    aliases, groups = list(res["runs"]), [*res["datasets"], "mean"]
    fig = Figure(figsize=(9, 4.6), facecolor=SURFACE)
    ax = fig.subplots()
    _style(ax)
    width = 0.8 / len(aliases)
    x = np.arange(len(groups))
    for i, a in enumerate(aliases):
        vals = [
            res["mean_of_datasets"][a]["miou_wo"]
            if g == "mean"
            else res["official"][a][g]["miou_wo"]
            for g in groups
        ]
        ivs = [res["bootstrap"]["miou_wo"][a][g] for g in groups]
        # The point is the official score; X4's interval comes from its recomputed
        # counts (M10), so a point may sit a hair outside it: clip, do not fail.
        err = np.clip(
            [[v - lo for v, (lo, _) in zip(vals, ivs)], [hi - v for v, (_, hi) in zip(vals, ivs)]],
            0,
            None,
        )
        pos = x + (i - (len(aliases) - 1) / 2) * width
        ax.bar(
            pos,
            vals,
            width * 0.92,
            color=RUN_COLOURS[i],
            edgecolor=SURFACE,
            lw=1,
            label=_label(res, a),
        )
        ax.errorbar(pos, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for p, v, (_, hi) in zip(pos, vals, ivs):
            ax.text(p, hi + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [g.upper() if g != "mean" else "mean of 3" for g in groups], color=INK)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"test mIoU (no {', '.join(res['exclude'])})", color=INK)
    _legend(ax, loc="upper left", ncols=len(aliases))
    lvl = f"{res['bootstrap']['level']:.0%}"
    ax.set_title(
        f"Test mIoU per test set; bars: {lvl} bootstrap interval over test images",
        loc="left",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout()
    path = Path(out) / "miou_by_test_set.png"
    fig.savefig(path, dpi=110)
    return path


def plot_per_class(res: Mapping, out: Path, metric: str = "iou") -> Path:
    """One panel per test set: per-class IoU (or Dice) of each run as grouped bars."""
    aliases, datasets = list(res["runs"]), list(res["datasets"])
    fig = Figure(figsize=(11, 3.4 * len(datasets) + 0.6), facecolor=SURFACE)
    axes = fig.subplots(len(datasets), 1, sharex=True)
    width = 0.8 / len(aliases)
    x = np.arange(NUM_CLASSES)
    for ax, d in zip(np.atleast_1d(axes), datasets):
        _style(ax)
        for i, a in enumerate(aliases):
            vals = [res["official"][a][d][metric][c] for c in SHARED_CLASSES]
            vals = [np.nan if v is None else v for v in vals]
            ax.bar(
                x + (i - (len(aliases) - 1) / 2) * width,
                vals,
                width * 0.92,
                color=RUN_COLOURS[i],
                edgecolor=SURFACE,
                lw=1,
                label=_label(res, a),
            )
        ax.set_ylim(0, 1)
        ax.set_ylabel(
            f"{d.upper()} test {metric.upper() if metric == 'iou' else 'Dice'}", color=INK
        )
        absent = [c for c in SHARED_CLASSES if c not in res["official"][aliases[0]][d]["in_mean"]]
        if absent:
            ax.text(
                0.99,
                0.95,
                "no ground truth: " + ", ".join(absent),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=MUTED,
            )
        # class names under every panel, not only the last (sharex hides them)
        ax.set_xticks(x, SHARED_CLASSES, rotation=30, ha="right", color=INK)
        ax.tick_params(axis="x", labelbottom=True)
    _legend(
        np.atleast_1d(axes)[0], loc="upper center", ncols=len(aliases), bbox_to_anchor=(0.5, 1.28)
    )
    fig.tight_layout()
    path = Path(out) / f"{metric}_per_class.png"
    fig.savefig(path, dpi=110)
    return path


def row_normalised(cm: np.ndarray) -> np.ndarray:
    """Each row / its sum (share of a class's ground truth per prediction); NaN for empty rows."""
    cm = np.asarray(cm, dtype=np.float64)
    rows = cm.sum(1, keepdims=True)
    return np.divide(cm, rows, out=np.full(cm.shape, np.nan), where=rows > 0)


def plot_confusion(
    official: Mapping[str, Mapping[str, Mapping]],
    res: Mapping,
    dataset: str,
    out: Path,
    min_label: float = 0.05,
) -> Path:
    """Row-normalised confusion matrices of every run on one test set, side by side."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_CMAP).with_extremes(bad=GRID)
    aliases = list(res["runs"])
    fig = Figure(figsize=(6.2 * len(aliases), 6.4), facecolor=SURFACE)
    axes = fig.subplots(1, len(aliases))
    for ax, a in zip(np.atleast_1d(axes), aliases):
        m = official[a][dataset]
        r = row_normalised(np.asarray(m["confusion_matrix"]))
        ax.imshow(r, cmap=cmap, vmin=0, vmax=1)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                v = r[i, j]
                if np.isfinite(v) and v >= min_label:
                    ax.text(
                        j,
                        i,
                        f"{v * 100:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if v > 0.55 else INK,
                    )
        names = list(m["classes"])
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8, color=INK)
        ax.set_yticks(range(len(names)), names, fontsize=8, color=INK)
        ax.set_xlabel("predicted", color=MUTED)
        ax.set_ylabel("ground truth", color=MUTED)
        ax.set_title(f"{a} {res['runs'][a]['model']}", color=INK, fontsize=11)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(
        f"{dataset.upper()} test: share of each true class's pixels per predicted class (%, "
        f"cells < {min_label:.0%} unlabelled; grey row = no ground truth)",
        x=0.01,
        ha="left",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    path = Path(out) / f"confusion_{dataset}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0.15)
    return path


def plot_differences(res: Mapping, out: Path) -> Path:
    """Paired mIoU differences between runs with their bootstrap intervals."""
    diff = res["bootstrap"]["diff"]
    groups = [*res["datasets"], "mean"]
    pairs = list(diff)
    fig = Figure(figsize=(9, 1.0 + 0.9 * len(pairs) * 1.3), facecolor=SURFACE)
    ax = fig.subplots()
    _style(ax)
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.axvline(0, color=MUTED, lw=1)
    step = 0.8 / len(groups)
    shapes = ("o", "s", "^", "D")
    for gi, g in enumerate(groups):
        ys = [pi + (gi - (len(groups) - 1) / 2) * step for pi in range(len(pairs))]
        vals = [diff[p][g]["diff"] for p in pairs]
        lo = [diff[p][g]["interval"][0] for p in pairs]
        hi = [diff[p][g]["interval"][1] for p in pairs]
        ax.hlines(ys, lo, hi, color=MUTED, lw=2)
        ax.plot(
            vals,
            ys,
            shapes[gi % len(shapes)],
            ms=8,
            color=INK if g == "mean" else MUTED,
            mec=SURFACE,
            mew=1.5,
            ls="none",
            label="mean of 3" if g == "mean" else g.upper(),
        )
        for v, y, h in zip(vals, ys, hi):
            ax.text(h + 0.002, y, f"{v:+.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(pairs)), pairs, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel(f"difference in test mIoU (no {', '.join(res['exclude'])})", color=INK)
    ax.legend(
        frameon=False, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.06), ncols=len(groups)
    )
    lvl = f"{res['bootstrap']['level']:.0%}"
    fig.suptitle(
        f"Paired differences; lines: {lvl} bootstrap interval over test images",
        x=0.01,
        ha="left",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout()
    path = Path(out) / "miou_differences.png"
    fig.savefig(path, dpi=110)
    return path
