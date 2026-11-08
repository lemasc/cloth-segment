"""Predict LIP/CIHP test sets and count wrong 'person' pixels on background (M11).

Needs the GPU lock. Writes per run and test set, under `data/<out_dir>/<run>/`:
`<dataset>.json` (summed confusion matrices: all pixels, then one per
blob-ignore threshold; check against the saved `metrics.json`) and
`<dataset>.csv` (counts per image, `cloth_segment.multiperson.image_result`).
For the review run and test set it also saves each prediction as a PNG under
`data/<review.out_dir>/pred/` for drawing. An existing JSON is never
overwritten (it is the cache). Same prediction path as `evaluate.py`, which is
not changed.

Usage: uv run python scripts/multiperson.py configs/report/m11.yaml [--wait]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from cloth_segment import errors, multiperson, paths, runs
from cloth_segment.data.dataset import DATASET_NAMES
from cloth_segment.data.labels import NUM_CLASSES
from cloth_segment.evaluate import (
    amp_dtype,
    eval_loader,
    load_eval_set,
    load_model,
    metrics_path,
    resolve_run,
)


def run_one(cfg: dict, alias: str, datasets: list[str], device) -> int:
    """Count one run on each of its test sets not done yet; returns check failures."""
    name = cfg["runs"][alias]["name"]
    run_dir = resolve_run(name)
    out = paths.DATA / cfg["out_dir"] / name
    todo = [d for d in datasets if not (out / f"{d}.json").exists()]
    if not todo:
        return 0
    rcfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model, step = load_model(run_dir, rcfg, device)
    dtype = amp_dtype(rcfg["train"]["amp"], device)
    workers = (rcfg.get("run") or {}).get("num_workers", 0)
    shares = cfg["blob_shares"]
    rv = cfg["review"]
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for ds in todo:
        t0 = time.perf_counter()
        data = load_eval_set(rcfg, ds, "test")
        ids = data.rows["id"].astype(str).tolist()
        save = None
        if (alias, ds) == (rv["run"], rv["dataset"]):
            save = paths.DATA / rv["out_dir"] / "pred" / name / ds
            save.mkdir(parents=True, exist_ok=True)
        loader = eval_loader(data, rcfg["train"]["batch_size"], workers)
        total = np.zeros((1 + len(shares), NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        rows = []
        it = errors.iter_predictions(model, loader, device, dtype)
        for i, (d_id, pred, gt) in enumerate(it):
            if DATASET_NAMES[d_id] != ds:
                raise ValueError(f"{ds}: loader returned an image of {DATASET_NAMES[d_id]}")
            stats, cms = multiperson.image_result(pred, gt, cfg["edge_frac"], shares)
            total += cms
            rows.append({"i": i, "id": ids[i], **stats})
            if save is not None:
                Image.fromarray(pred).save(save / f"{ids[i]}.png")
        n = len(rows)
        saved = json.loads(metrics_path(run_dir, ds).read_text())
        diff = int(np.abs(total[0] - np.asarray(saved["confusion_matrix"])).max())
        failures += int(n != saved["images"] or total[0].sum() != np.sum(saved["confusion_matrix"]))
        pd.DataFrame(rows).to_csv(out / f"{ds}.csv", index=False)
        result = {
            "run": name,
            "dataset": ds,
            "images": n,
            "checkpoint_step": step,
            "edge_frac": cfg["edge_frac"],
            "blob_shares": shares,
            # [0] all pixels, [1 + i] blob-ignore at blob_shares[i]; 12x12, rows = ground truth
            "confusion": total.tolist(),
            "check_vs_saved_metrics": {
                "metrics_file": str(metrics_path(run_dir, ds).relative_to(run_dir)),
                "same_matrix": diff == 0,
                "max_abs_diff": diff,
            },
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "seconds": round(time.perf_counter() - t0, 1),
        }
        tmp = out / f"{ds}.json.tmp"
        tmp.write_text(json.dumps(result) + "\n")
        tmp.replace(out / f"{ds}.json")
        secs = result["seconds"]
        print(f"{name} {ds}: {n} images, max diff vs saved {diff}, {secs} s", flush=True)
    del model
    torch.cuda.empty_cache()
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--wait", action="store_true", help="wait for the GPU lock")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    by_run: dict[str, list[str]] = {}
    for ds, aliases in cfg["jobs"].items():
        for a in aliases:
            by_run.setdefault(a, []).append(ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failures = 0
    try:
        with runs.gpu_lock(run="m11-multiperson", wait=args.wait):
            for alias, datasets in by_run.items():
                failures += run_one(cfg, alias, datasets, device)
    except runs.GpuBusy as e:
        print(e, file=sys.stderr)
        return 2
    if failures:
        print(f"{failures} test sets do not cover the saved metrics' pixels", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
