"""T10 error analysis from saved files: error zones, scores by image size and
shape, and the CIHP `dress` review (analysis only, no GPU).

Inputs (read only):
- `data/<zones.out_dir>/<run>/<dataset>.json` from `scripts/error_zones.py`;
- `data/<breakdown_dir>/<run>/<dataset>/per_image_shared.csv` (M10);
- the review samples (`data/<out_dir>/samples.csv`) and verdicts CSV (T10 review).

Bucket scores add up the per-image counts of the images in a bucket and score
them like a test set (`results.miou_from_counts`), so each bucket's mIoU is a
dataset-level score, not a mean of per-image scores.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cloth_segment import errors, results
from cloth_segment.data.labels import SHARED_CLASSES

# Categorical slots 7, 4, 5 of the dataviz reference palette (validated, light
# mode); slots 1-3 are taken by the runs in the T9 figures.
ZONE_COLOURS = {"edge": "#6250d6", "swap": "#eda100", "inner": "#e87ba4"}
ZONE_LABELS = {"edge": "edge band", "swap": "swapped region", "inner": "other interior"}
BG = SHARED_CLASSES.index("background")


# ---------------------------------------------------------------- error zones


def read_zones(path: Path) -> dict:
    """One `error_zones.py` JSON, with the zone matrices as an int64 array."""
    z = json.loads(Path(path).read_text())
    z["zone_confusion"] = np.asarray(z["zone_confusion"], dtype=np.int64)
    if list(z["zones"]) != list(errors.ZONES):
        raise ValueError(f"{path}: zones {z['zones']}, expected {errors.ZONES}")
    return z


def zone_summary(z: Mapping, headline: float, exclude: Sequence[str], top: int = 5) -> dict:
    """Error shares per band width; per-class shares, mIoU without the edge band
    and the largest swapped pairs at the `headline` width."""
    fracs = list(z["fracs"])
    if headline not in fracs:
        raise ValueError(f"headline width {headline} not in {fracs}")
    zc = z["zone_confusion"]
    h = zc[fracs.index(headline)]
    full = h.sum(0)
    wrong = int(full.sum() - np.trace(full))
    no_edge = results.score_cm(h[1] + h[2])
    swap = h[1].copy()
    np.fill_diagonal(swap, 0)
    pairs = sorted(
        (
            (int(swap[i, j]), SHARED_CLASSES[i], SHARED_CLASSES[j])
            for i in range(len(SHARED_CLASSES))
            for j in range(len(SHARED_CLASSES))
            if swap[i, j]
        ),
        reverse=True,
    )[:top]
    per_class = errors.class_error_shares(h)
    return {
        "images": z["images"],
        "wrong_px": wrong,
        "shares": {str(f): errors.error_shares(zc[i]) for i, f in enumerate(fracs)},
        "class_shares": {
            c: {
                zn: (None if np.isnan(per_class[i, k]) else float(per_class[i, k]))
                for k, zn in enumerate(errors.ZONES)
            }
            for i, c in enumerate(SHARED_CLASSES)
        },
        "miou_wo": results.mean_without(results.score_cm(full), "iou", exclude),
        "miou_wo_no_edge": results.mean_without(no_edge, "iou", exclude),
        "top_swaps": [
            {"gt": g, "pred": p, "px": n, "share_of_wrong": n / wrong} for n, g, p in pairs
        ],
        "check": z["check_vs_saved_metrics"],
    }


# ---------------------------------------------------------------- buckets


def read_image_table(path: Path) -> pd.DataFrame:
    """Per image: index, h, w from `per_image_shared.csv` (one row per image)."""
    df = pd.read_csv(path, usecols=["index", "h", "w"])
    return df.drop_duplicates("index").set_index("index").sort_index()


def person_share(counts: np.ndarray) -> np.ndarray:
    """(images,) share of scored pixels that are not background (how big the people are)."""
    gt = counts[:, 0, :]
    tot = gt.sum(1)
    return np.divide(tot - gt[:, BG], tot, out=np.zeros(len(gt)), where=tot > 0)


def bucket_scores(
    counts: np.ndarray,
    values: np.ndarray,
    edges: Sequence[float],
    labels: Sequence[str],
    exclude: Sequence[str],
) -> list[dict]:
    """mIoU (without `exclude`) of the images in each [edge_i, edge_i+1) bucket of `values`."""
    if len(labels) != len(edges) - 1:
        raise ValueError("need one label per bucket")
    if len(values) != len(counts):
        raise ValueError("one value per image")
    keep = results.keep_mask(exclude)
    idx = np.digitize(values, edges[1:-1], right=False)
    out = []
    for b, lab in enumerate(labels):
        sel = idx == b
        n = int(sel.sum())
        m = results.miou_from_counts(counts[sel].sum(0), keep) if n else np.nan
        out.append({"bucket": lab, "images": n, "miou_wo": None if np.isnan(m) else float(m)})
    return out


# ---------------------------------------------------------------- report


def build(
    cfg: Mapping,
    zones: Mapping[str, Mapping[str, Mapping]],
    counts: Mapping[str, Mapping[str, np.ndarray]],
    sizes: Mapping[str, pd.DataFrame],
    review: Mapping | None,
) -> dict:
    """Every T10 number. `zones`/`counts` keyed by run alias then dataset;
    `sizes` by dataset; `review` = {"samples", "verdicts", "codes"} or None."""
    aliases, datasets = list(cfg["runs"]), list(cfg["datasets"])
    exclude = list(cfg["exclude"])
    zc = cfg["zones"]
    res: dict = {"runs": {a: cfg["runs"][a] for a in aliases}, "datasets": datasets}
    res["exclude"] = exclude
    res["zones"] = {"fracs": zc["fracs"], "headline": zc["headline"], "per_run": {}}
    for a in aliases:
        res["zones"]["per_run"][a] = {
            d: zone_summary(zones[a][d], zc["headline"], exclude) for d in datasets
        }
    res["buckets"] = {}
    for key, spec in cfg["buckets"].items():
        res["buckets"][key] = {"title": spec["title"], "per_dataset": {}}
        for d in spec["datasets"]:
            t = sizes[d]
            if spec["value"] == "short_side":
                vals = np.minimum(t["h"], t["w"]).to_numpy()
            elif spec["value"] == "aspect":
                vals = (t["h"] / t["w"]).to_numpy()
            elif spec["value"] == "person_share":
                vals = person_share(counts[aliases[-1]][d])
            else:
                raise ValueError(f"unknown bucket value {spec['value']!r}")
            res["buckets"][key]["per_dataset"][d] = {
                a: bucket_scores(counts[a][d], vals, spec["edges"], spec["labels"], exclude)
                for a in aliases
            }
    if review is not None:
        res["review"] = review_summary(review["samples"], review["verdicts"], review["codes"])
    return res


def review_summary(samples: pd.DataFrame, verdicts: pd.DataFrame, codes: Mapping) -> dict:
    """Verdict counts per pick group, and how many images are still without a verdict."""
    v = verdicts[verdicts["key"].isin(samples["key"]) & (verdicts["verdict"] != "")]
    m = samples[["key", "pick"]].merge(v[["key", "verdict"]], on="key", how="left")
    m["verdict"] = m["verdict"].fillna("")
    out = {"images": len(samples), "reviewed": int((m["verdict"] != "").sum()), "counts": {}}
    for code in codes:
        out["counts"][code] = {
            p: int(((m["verdict"] == code) & (m["pick"] == p)).sum()) for p in ("worst", "random")
        }
    return out


def _f(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def _p(v):
    return "-" if v is None else f"{v:.1%}"


def tables_markdown(res: Mapping) -> str:
    """All T10 tables, every value from `res`."""
    aliases, datasets = list(res["runs"]), list(res["datasets"])
    zr = res["zones"]
    head = zr["headline"]
    wo = ", ".join(res["exclude"])
    md = ["# T10 tables (generated by scripts/t10_results.py; do not edit by hand)\n"]

    md += [f"## Wrong pixels by zone (edge band = {head:.1%} of the image diagonal)\n"]
    md += ["| test set | run | wrong px | edge | swapped region | other interior |"]
    md.append("|---|---|---|---|---|---|")
    for d in datasets:
        for a in aliases:
            s = zr["per_run"][a][d]
            sh = s["shares"][str(head)]
            md.append(
                f"| {d} | {a} | {s['wrong_px']:,} | {_p(sh['edge'])} | {_p(sh['swap'])} "
                f"| {_p(sh['inner'])} |"
            )

    md += ["\n## Edge share for every band width (robustness)\n"]
    md += ["| test set | run | " + " | ".join(f"edge @ {f:.1%}" for f in zr["fracs"]) + " |"]
    md.append("|---|---|" + "---|" * len(zr["fracs"]))
    for d in datasets:
        for a in aliases:
            sh = zr["per_run"][a][d]["shares"]
            md.append(
                f"| {d} | {a} | " + " | ".join(_p(sh[str(f)]["edge"]) for f in zr["fracs"]) + " |"
            )

    md += [f"\n## mIoU (no {wo}) with and without the edge band\n"]
    md += ["| test set | run | all pixels | edge band left out | gain |", "|---|---|---|---|---|"]
    for d in datasets:
        for a in aliases:
            s = zr["per_run"][a][d]
            md.append(
                f"| {d} | {a} | {_f(s['miou_wo'])} | {_f(s['miou_wo_no_edge'])} "
                f"| {s['miou_wo_no_edge'] - s['miou_wo']:+.4f} |"
            )

    md += ["\n## Largest swapped-region errors (share of all wrong pixels)\n"]
    md += ["| test set | run | " + " | ".join(f"#{i + 1}" for i in range(5)) + " |"]
    md.append("|---|---|" + "---|" * 5)
    for d in datasets:
        for a in aliases:
            sw = zr["per_run"][a][d]["top_swaps"]
            cells = [f"{s['gt']} → {s['pred']} {s['share_of_wrong']:.1%}" for s in sw]
            cells += ["-"] * (5 - len(cells))
            md.append(f"| {d} | {a} | " + " | ".join(cells) + " |")

    last = aliases[-1]
    md += [f"\n## Per class: where its wrong pixels are ({last}, edge band {head:.1%})\n"]
    md += ["| class | " + " | ".join(f"{d} edge / swap / inner" for d in datasets) + " |"]
    md.append("|---|" + "---|" * len(datasets))
    for c in SHARED_CLASSES:
        cells = []
        for d in datasets:
            cs = zr["per_run"][last][d]["class_shares"][c]
            cells.append(" / ".join(_p(cs[z]) for z in errors.ZONES))
        md.append(f"| {c} | " + " | ".join(cells) + " |")

    for b in res["buckets"].values():
        md += [f"\n## mIoU (no {wo}) by {b['title']}\n"]
        for d, per in b["per_dataset"].items():
            labels = [r["bucket"] for r in per[aliases[0]]]
            md += [f"**{d}**\n", "| run | " + " | ".join(labels) + " |"]
            md.append("|---|" + "---|" * len(labels))
            md.append("| images | " + " | ".join(str(r["images"]) for r in per[aliases[0]]) + " |")
            for a in aliases:
                md.append(f"| {a} | " + " | ".join(_f(r["miou_wo"]) for r in per[a]) + " |")
            md.append("")

    if "review" in res:
        r = res["review"]
        md += [f"\n## CIHP dress review: {r['reviewed']} of {r['images']} images judged\n"]
        md += ["| verdict | worst | random |", "|---|---|---|"]
        for code, c in r["counts"].items():
            md.append(f"| {code} | {c['worst']} | {c['random']} |")

    checks = [
        f"| {a} | {d} | {zr['per_run'][a][d]['images']} | "
        f"{'yes' if zr['per_run'][a][d]['check']['same_matrix'] else 'no'} | "
        f"{zr['per_run'][a][d]['check']['max_abs_diff']} |"
        for a in aliases
        for d in datasets
    ]
    md += ["\n## Check: zones add up to the saved metrics.json matrix\n"]
    md += ["| run | test set | images | same matrix | max abs cell diff |", "|---|---|---|---|---|"]
    md += checks
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- figures

INK, MUTED, GRID, SURFACE = results.INK, results.MUTED, results.GRID, results.SURFACE


def plot_zone_shares(res: Mapping, out: Path) -> Path:
    """100% stacked bars: zone shares of the wrong pixels, one bar per test set and run."""
    aliases, datasets = list(res["runs"]), list(res["datasets"])
    head = str(res["zones"]["headline"])
    labels, shares = [], []
    for d in datasets:
        for a in aliases:
            labels.append(f"{d.upper()} {a}")
            shares.append(res["zones"]["per_run"][a][d]["shares"][head])
    fig = Figure(figsize=(9, 0.42 * len(labels) + 1.4), facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for z in errors.ZONES:
        v = np.array([s[z] or 0 for s in shares])
        ax.barh(
            y,
            v,
            left=left,
            color=ZONE_COLOURS[z],
            edgecolor=SURFACE,
            lw=2,
            label=ZONE_LABELS[z],
            height=0.72,
        )
        for yy, l0, vv in zip(y, left, v, strict=True):
            if vv >= 0.06:
                ax.text(
                    l0 + vv / 2,
                    yy,
                    f"{vv:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if z == "edge" else INK,
                )
        left += v
    ax.set_yticks(y, labels, color=INK, fontsize=9)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, labelcolor=INK, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.suptitle(
        f"Where the wrong pixels are (edge band = {float(head):.1%} of the image diagonal)",
        x=0.01,
        ha="left",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout()
    path = Path(out) / "error_zones.png"
    fig.savefig(path, dpi=110)
    return path


def plot_class_zones(res: Mapping, out: Path) -> Path:
    """Per test set, per class: zone shares of the last run's wrong pixels (stacked bars)."""
    datasets = list(res["datasets"])
    last = list(res["runs"])[-1]
    fig = Figure(figsize=(11, 2.6 * len(datasets) + 0.8), facecolor=SURFACE)
    axes = np.atleast_1d(fig.subplots(len(datasets), 1, sharex=True))
    x = np.arange(len(SHARED_CLASSES))
    for ax, d in zip(axes, datasets, strict=True):
        ax.set_facecolor(SURFACE)
        cs = res["zones"]["per_run"][last][d]["class_shares"]
        bottom = np.zeros(len(x))
        for z in errors.ZONES:
            v = np.array([cs[c][z] or 0 for c in SHARED_CLASSES])
            ax.bar(
                x,
                v,
                bottom=bottom,
                color=ZONE_COLOURS[z],
                edgecolor=SURFACE,
                lw=1.5,
                width=0.75,
                label=ZONE_LABELS[z],
            )
            bottom += v
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_ylabel(f"{d.upper()}", color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    axes[-1].set_xticks(x, SHARED_CLASSES, rotation=30, ha="right", color=INK)
    axes[0].legend(
        frameon=False, labelcolor=INK, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.02)
    )
    fig.suptitle(
        f"{last} {res['runs'][last]['model']}: where each class's wrong pixels are",
        x=0.01,
        ha="left",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout()
    path = Path(out) / "error_zones_per_class.png"
    fig.savefig(path, dpi=110)
    return path


def plot_buckets(res: Mapping, key: str, out: Path) -> Path:
    """mIoU per bucket, one panel per test set, one line per run."""
    b = res["buckets"][key]
    aliases = list(res["runs"])
    per = b["per_dataset"]
    fig = Figure(figsize=(4.2 * len(per), 3.8), facecolor=SURFACE)
    axes = np.atleast_1d(fig.subplots(1, len(per), sharey=True))
    for ax, (d, runs_) in zip(axes, per.items(), strict=True):
        results._style(ax)
        labels = [r["bucket"] for r in runs_[aliases[0]]]
        x = np.arange(len(labels))
        for i, a in enumerate(aliases):
            ys = [np.nan if r["miou_wo"] is None else r["miou_wo"] for r in runs_[a]]
            ax.plot(
                x,
                ys,
                marker="o",
                ms=7,
                lw=2,
                color=results.RUN_COLOURS[i],
                mec=SURFACE,
                mew=1.5,
                label=f"{a} {res['runs'][a]['model']}",
            )
        n = [r["images"] for r in runs_[aliases[0]]]
        ax.set_xticks(
            x, [f"{lab}\n(n={k})" for lab, k in zip(labels, n, strict=True)], fontsize=8, color=INK
        )
        ax.set_title(d.upper(), color=INK, fontsize=10)
    axes[0].set_ylabel(f"test mIoU (no {', '.join(res['exclude'])})", color=INK)
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=8, loc="lower right")
    fig.suptitle(f"mIoU by {b['title']}", x=0.01, ha="left", color=MUTED, fontsize=10)
    fig.tight_layout()
    path = Path(out) / f"buckets_{key}.png"
    fig.savefig(path, dpi=110)
    return path
