"""Per-raw-class and per-image breakdown of finished runs on the test sets. Needs the GPU lock.

Generic version of `accessory_by_raw_class.py`: covers every class. For each
run and test set it writes, under OUT/<run>/<dataset>/:

- `raw_confusion.json`: for every raw class, its pixels, the shared class it
  maps to, recall (pixels predicted as that shared class) and where its pixels
  went (predicted shared class -> pixels); plus a cross-check of the summed
  per-image counts against the run's saved `metrics.json`;
- `per_image_shared.csv`: per image and shared class (present in the ground
  truth or the prediction): gt, pred and true-positive pixels, image H and W;
- `per_image_raw.csv`: per image and raw class present: raw pixels and pixels
  predicted as its shared class.

Read-only on runs/ and data/. Pixel counts only, no images.

Usage: uv run python scripts/raw_class_breakdown.py RUN [RUN ...] --out-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from cloth_segment import runs
from cloth_segment.data.labels import NUM_CLASSES, SHARED_CLASSES
from cloth_segment.evaluate import amp_dtype, load_eval_set, load_model, metrics_path, resolve_run
from cloth_segment.rawclass import (
    RAW_MAX,
    iter_predictions,
    raw_loader,
    raw_pred_matrix,
    raw_rows,
    shared_counts,
)


def breakdown(model, data, cfg, device, dtype, dataset: str, out: Path) -> dict:
    workers = (cfg.get("run") or {}).get("num_workers", 0)
    loader = raw_loader(data, cfg["train"]["batch_size"], workers)
    ids = data.rows["id"].astype(str).tolist()
    total_raw = torch.zeros(RAW_MAX, NUM_CLASSES, dtype=torch.int64)
    total_shared = torch.zeros(3, NUM_CLASSES, dtype=torch.int64)
    out.mkdir(parents=True, exist_ok=True)
    with (
        open(out / "per_image_shared.csv", "w", newline="") as fs,
        open(out / "per_image_raw.csv", "w", newline="") as fr,
    ):
        ws = csv.writer(fs)
        wr = csv.writer(fr)
        ws.writerow(["index", "id", "h", "w", "cls", "gt", "pred", "tp"])
        wr.writerow(["index", "id", "raw_id", "raw_name", "shared", "raw_px", "hit_px"])
        for idx, pred, target, raw in iter_predictions(model, loader, device, dtype):
            sc = shared_counts(pred, target).cpu()
            rm = raw_pred_matrix(pred, raw).cpu()
            total_shared += sc
            total_raw += rm
            h, w = target.shape
            for c in range(NUM_CLASSES):
                if sc[0, c] or sc[1, c]:
                    ws.writerow([idx, ids[idx], h, w, SHARED_CLASSES[c], *sc[:, c].tolist()])
            for r in raw_rows(rm, dataset):
                wr.writerow(
                    [
                        idx,
                        ids[idx],
                        r["raw_id"],
                        r["raw_name"],
                        r["shared"],
                        r["raw_px"],
                        r["hit_px"],
                    ]
                )
    raw = {}
    for r in raw_rows(total_raw, dataset):
        d = total_raw[r["raw_id"]]
        raw[r["raw_name"]] = r | {
            "recall": r["hit_px"] / r["raw_px"] if r["hit_px"] is not None else None,
            "where": {c: int(d[i]) for i, c in enumerate(SHARED_CLASSES) if int(d[i])},
        }
    return {"images": len(data), "raw_classes": raw, "shared_totals": total_shared}


def check(run_dir: Path, dataset: str, totals: torch.Tensor) -> dict:
    """Compare summed per-image counts with the saved confusion matrix."""
    path = metrics_path(run_dir, dataset)
    if not path.exists():
        return {"metrics_file": None}
    cm = torch.tensor(json.loads(path.read_text())["confusion_matrix"], dtype=torch.int64)
    saved = torch.stack([cm.sum(1), cm.sum(0), cm.diagonal()])
    return {
        "metrics_file": str(path.relative_to(run_dir)),
        "match": bool(torch.equal(saved, totals)),
        "max_abs_diff": int((saved - totals).abs().max()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--datasets", nargs="+", default=["lip", "cihp", "dfmm"])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--wait", action="store_true", help="wait for the GPU lock")
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failed = False
    try:
        with runs.gpu_lock(run="raw-class-breakdown", wait=args.wait):
            for name in args.runs:
                run_dir = resolve_run(name)
                cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
                model, step = load_model(run_dir, cfg, device)
                dtype = amp_dtype(cfg["train"]["amp"], device)
                for ds in args.datasets:
                    t0 = time.perf_counter()
                    out = args.out_dir / run_dir.name / ds
                    data = load_eval_set(cfg, ds, "test")
                    res = breakdown(model, data, cfg, device, dtype, ds, out)
                    totals = res.pop("shared_totals")
                    res = {"run": run_dir.name, "dataset": ds, "checkpoint_step": step} | res
                    res["check_vs_saved_metrics"] = chk = check(run_dir, ds, totals)
                    res["seconds"] = round(time.perf_counter() - t0, 1)
                    (out / "raw_confusion.json").write_text(json.dumps(res, indent=2) + "\n")
                    failed |= chk.get("match") is False
                    print(f"{run_dir.name} {ds}: {res['seconds']}s check={chk}", flush=True)
    except runs.GpuBusy as e:
        print(e, file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
