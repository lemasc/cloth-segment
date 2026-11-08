"""M11 numbers, review sample and sheets from the files `scripts/multiperson.py` wrote. No GPU.

Writes the results JSON and tables Markdown named in the config (committed),
and `data/<review.out_dir>/samples.csv` (the images to look at). With
`--sheets` it also draws image | GT | prediction | wrong-person overlay sheets
to `data/<review.out_dir>/sheets/` (dataset images: never committed). Verdicts
are tallied when the verdict file exists.

Usage: uv run python scripts/m11_results.py configs/report/m11.yaml [--sheets]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

from cloth_segment import errors, multiperson, paths
from cloth_segment.data.dataset import read_mask
from cloth_segment.results import mean_without, score_cm
from cloth_segment.review import colour_mask


def job_summary(cfg: dict, name: str, ds: str) -> dict:
    """Counts, concentration and analysis scores of one run on one test set."""
    base = paths.DATA / cfg["out_dir"] / name
    res = json.loads((base / f"{ds}.json").read_text())
    st = pd.read_csv(base / f"{ds}.csv", dtype={"id": str})
    cms = np.asarray(res["confusion"])
    exclude = cfg["exclude"]
    official = score_cm(cms[0])
    top10 = st.sort_values("fp_int", ascending=False).head(int(round(len(st) * 0.1)))
    out = {
        "images": len(st),
        "check_vs_saved_metrics": res["check_vs_saved_metrics"],
        "device": res["device"],
        "bg_pixels": int(st["bg"].sum()),
        "fp_all": int(st["fp_all"].sum()),
        "fp_int": int(st["fp_int"].sum()),
        "fp_int_per_bg": float(st["fp_int"].sum() / st["bg"].sum()),
        "images_with_fp_int": int((st["fp_int"] > 0).sum()),
        "fp_int_separate_share": float(st["fp_int_sep"].sum() / max(st["fp_int"].sum(), 1)),
        "concentration": multiperson.concentration(st["fp_int"].to_numpy(), cfg["tops"]),
        "median_person_share": {
            "all": float(st["person_share"].median()),
            "top10pct_fp_int": float(top10["person_share"].median()),
        },
        "miou": mean_without(official, "iou", exclude),
        "iou": official["iou"],
        "blob_ignore": {},
    }
    for k, s in enumerate(res["blob_shares"]):
        sc = score_cm(cms[1 + k])
        out["blob_ignore"][str(s)] = {
            "miou": mean_without(sc, "iou", exclude),
            "iou": sc["iou"],
            "images_affected": int((st[f"ignored_{s}"] > 0).sum()),
            "ignored_pixels": int(st[f"ignored_{s}"].sum()),
            "fp_int_ignored_share": float(
                st[f"fp_int_ignored_{s}"].sum() / max(st["fp_int"].sum(), 1)
            ),
        }
    return out


def sample(cfg: dict) -> pd.DataFrame:
    """The review sample; writes it, and refuses to replace a different one."""
    rv = cfg["review"]
    name = cfg["runs"][rv["run"]]["name"]
    st = pd.read_csv(paths.DATA / cfg["out_dir"] / name / f"{rv['dataset']}.csv", dtype={"id": str})
    s = multiperson.pick_review(st, rv["top"], rv["random"], rv["random_min_px"], rv["seed"])
    path = paths.DATA / rv["out_dir"] / "samples.csv"
    if path.exists():
        old = pd.read_csv(path, dtype={"id": str})
        if old["id"].tolist() != s["id"].tolist():
            raise SystemExit(f"{path} holds another sample; change review.out_dir to draw again")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(path, index=False)
    return s


def tile(cfg: dict, row, manifest: pd.DataFrame) -> np.ndarray:
    """image | GT | prediction | interior false-person pixels in magenta, `tile` px high."""
    rv = cfg["review"]
    name = cfg["runs"][rv["run"]]["name"]
    m = manifest.loc[row.id]
    img = np.asarray(Image.open(paths.DATASETS / m["image"]).convert("RGB"))
    gt = read_mask(paths.DATA / m["mask"]).numpy()
    pred = np.asarray(
        Image.open(paths.DATA / rv["out_dir"] / "pred" / name / rv["dataset"] / f"{row.id}.png")
    )
    h, w = gt.shape
    _, fp_int = multiperson.false_person(pred, gt, errors.band_px(h, w, cfg["edge_frac"]))
    ov = img.copy()
    ov[fp_int] = (0.4 * ov[fp_int] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
    size = (max(1, round(w * rv["tile"] / h)), rv["tile"])
    parts = [
        np.asarray(Image.fromarray(a).resize(size, Image.NEAREST))
        for a in (img, colour_mask(gt), colour_mask(pred), ov)
    ]
    gap = np.full((rv["tile"], 5, 3), 255, np.uint8)
    body = np.concatenate([x for p in parts for x in (p, gap)][:-1], axis=1)
    head = Image.new("RGB", (body.shape[1], 16), "white")
    share = row.fp_int / (h * w)
    label = f"#{row.n} {row.group} {row.id} interior wrong-person px {row.fp_int} ({share:.1%})"
    ImageDraw.Draw(head).text((2, 2), label, fill="black")
    return np.concatenate([np.asarray(head), body], axis=0)


def sheets(cfg: dict, s: pd.DataFrame) -> list[Path]:
    """Two columns of tiles per sheet, `per_sheet` tiles each."""
    rv = cfg["review"]
    man = pd.read_csv(paths.DATA / "splits/splits-v2/manifest.csv", dtype={"id": str})
    man = man[(man["dataset"] == rv["dataset"]) & (man["split"] == "test")].set_index("id")
    out = paths.DATA / rv["out_dir"] / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for k in range(0, len(s), rv["per_sheet"]):
        tiles = [tile(cfg, r, man) for r in s.iloc[k : k + rv["per_sheet"]].itertuples()]
        width = max(t.shape[1] for t in tiles)
        tiles = [
            np.pad(t, ((0, 10), (0, width - t.shape[1] + 10), (0, 0)), constant_values=255)
            for t in tiles
        ]
        cols = [np.concatenate(tiles[j::2], axis=0) for j in range(min(2, len(tiles)))]
        height = max(c.shape[0] for c in cols)
        cols = [
            np.pad(c, ((0, height - c.shape[0]), (0, 0), (0, 0)), constant_values=255) for c in cols
        ]
        path = out / f"sheet_{k // rv['per_sheet']}.png"
        Image.fromarray(np.concatenate(cols, axis=1)).save(path)
        written.append(path)
    return written


def _f(v, nd: int = 4) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _rule(n: int) -> str:
    return "|" + "---|" * n


def tables(res: dict) -> str:
    """Markdown tables of everything in the results JSON."""
    head, shares, tops = res["headline_share"], res["blob_shares"], res["tops"]
    jobs = list(res["jobs"].values())
    L = ["# M11 tables (generated by `scripts/m11_results.py`; do not edit)", ""]

    L += ["## Counts (interior = outside the 1% edge band)", ""]
    cols = ["run", "test set", "images", "GT bg px", "wrong-person px on bg", "of which interior"]
    cols += ["interior / GT bg", "images with any", "separate-component share"]
    cols += ["matrix = metrics.json (max diff)"]
    L += [_row(*cols), _rule(len(cols))]
    for j in jobs:
        chk = j["check_vs_saved_metrics"]
        L += [
            _row(
                j["alias"],
                j["dataset"],
                j["images"],
                f"{j['bg_pixels']:,}",
                f"{j['fp_all']:,}",
                f"{j['fp_int']:,}",
                f"{j['fp_int_per_bg']:.4f}",
                j["images_with_fp_int"],
                f"{j['fp_int_separate_share']:.3f}",
                f"{chk['same_matrix']} ({chk['max_abs_diff']})",
            )
        ]

    L += ["", "## Concentration: share of interior wrong-person pixels in the top images", ""]
    cols = ["run", "test set", *(f"top {t:.0%}" for t in tops)]
    cols += ["median person share: all", "top 10%"]
    L += [_row(*cols), _rule(len(cols))]
    for j in jobs:
        c, mp = j["concentration"], j["median_person_share"]
        conc = [f"{c[str(t)]:.3f}" for t in tops]
        L += [
            _row(
                j["alias"], j["dataset"], *conc, f"{mp['all']:.3f}", f"{mp['top10pct_fp_int']:.3f}"
            )
        ]

    L += ["", "## mIoU (11 classes, no accessory) with big wrong-person blobs ignored", ""]
    L += [
        "Analysis score, not official: the rule also drops objects and unlabelled main people.",
        "",
    ]
    cols = ["run", "test set", "official", *(f"blob >= {s:.1%}" for s in shares)]
    cols += [f"images affected ({head:.0%})", f"interior px ignored ({head:.0%})"]
    L += [_row(*cols), _rule(len(cols))]
    for j in jobs:
        b = j["blob_ignore"]
        adj = [_f(b[str(s)]["miou"]) for s in shares]
        h = b[str(head)]
        L += [
            _row(
                j["alias"],
                j["dataset"],
                _f(j["miou"]),
                *adj,
                h["images_affected"],
                f"{h['fp_int_ignored_share']:.3f}",
            )
        ]

    L += ["", f"## Per-class IoU, official vs blob >= {head:.0%}", ""]
    classes = list(jobs[0]["iou"])
    L += [_row("run", "test set", *classes), _rule(len(classes) + 2)]
    for j in jobs:
        for tag, iou in (("official", j["iou"]), ("ignored", j["blob_ignore"][str(head)]["iou"])):
            L += [_row(j["alias"], f"{j['dataset']} {tag}", *(_f(iou[c], 3) for c in classes))]

    if "verdicts" in res:
        v = res["verdicts"]
        L += ["", f"## Visual check ({v['run']} on {v['dataset']}, {v['images']} images)", ""]
        cols = ["verdict", "top", "random", "total", "interior wrong-person px"]
        L += [_row(*cols), _rule(len(cols))]
        for r in v["tally"]:
            L += [_row(r["label"], r["top"], r["random"], r["total"], f"{r['fp_int']:,}")]
        sh = v["share_of_all_fp_int"]
        L += [
            "",
            f"Share of all interior wrong-person pixels of the test set: top group "
            f"{sh['top']:.3f}, random group {sh['random']:.3f}.",
        ]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--sheets", action="store_true", help="draw review sheets (dataset images)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    res = {
        "config": str(args.config),
        "edge_frac": cfg["edge_frac"],
        "blob_shares": cfg["blob_shares"],
        "headline_share": cfg["headline_share"],
        "tops": cfg["tops"],
        "exclude": cfg["exclude"],
        "jobs": {},
    }
    for ds, aliases in cfg["jobs"].items():
        for a in aliases:
            name = cfg["runs"][a]["name"]
            res["jobs"][f"{a}/{ds}"] = {
                "alias": a,
                "run": name,
                "dataset": ds,
                **job_summary(cfg, name, ds),
            }
    s = sample(cfg)
    if args.sheets:
        for p in sheets(cfg, s):
            print(p)
    vpath = paths.REPO_ROOT / cfg["verdicts"]
    if vpath.exists():
        codes = cfg["verdict_codes"]
        t = multiperson.tally_verdicts(pd.read_csv(vpath, dtype={"id": str}), s, list(codes))
        total = res["jobs"][f"{cfg['review']['run']}/{cfg['review']['dataset']}"]["fp_int"]
        res["verdicts"] = {
            "file": cfg["verdicts"],
            "run": cfg["review"]["run"],
            "dataset": cfg["review"]["dataset"],
            "images": len(s),
            "tally": [{"label": codes[r["verdict"]], **r} for r in t.to_dict("records")],
            "share_of_all_fp_int": {
                g: float(s.loc[s["group"] == g, "fp_int"].sum() / total) for g in ("top", "random")
            },
        }
    out = cfg["out"]
    (paths.REPO_ROOT / out["json"]).write_text(json.dumps(res, indent=1) + "\n")
    (paths.REPO_ROOT / out["tables"]).write_text(tables(res))
    print(out["json"], out["tables"], sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
