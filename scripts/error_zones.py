"""Score each run's `best.pt` on the test sets, splitting every wrong pixel into
edge / swapped region / inner (`cloth_segment.errors`). Needs the GPU lock.

Writes `data/<zones.out_dir>/<run>/<dataset>.json` per run and test set: the
zone confusion matrices for every band width, image count, and a check
against the saved `metrics.json`. An existing file is never overwritten (it is
the cache). Same prediction path as `evaluate.py`, which is not changed.

Usage: uv run python scripts/error_zones.py configs/report/t10.yaml [--wait]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from cloth_segment import errors, paths, runs
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


def zone_run(run_dir: Path, datasets: list[str], fracs: list[float], out: Path, device) -> int:
    """Zone matrices of one run on each test set not done yet; returns check failures."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    todo = [d for d in datasets if not (out / f"{d}.json").exists()]
    if not todo:
        return 0
    model, step = load_model(run_dir, cfg, device)
    dtype = amp_dtype(cfg["train"]["amp"], device)
    workers = (cfg.get("run") or {}).get("num_workers", 0)
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for ds in todo:
        t0 = time.perf_counter()
        data = load_eval_set(cfg, ds, "test")
        loader = eval_loader(data, cfg["train"]["batch_size"], workers)
        total = np.zeros((len(fracs), len(errors.ZONES), NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        n = 0
        for d_id, pred, gt in errors.iter_predictions(model, loader, device, dtype):
            if DATASET_NAMES[d_id] != ds:
                raise ValueError(f"{ds}: loader returned an image of {DATASET_NAMES[d_id]}")
            total += errors.zone_confusions(pred, gt, fracs)
            n += 1
        saved = json.loads(metrics_path(run_dir, ds).read_text())
        full = total[0].sum(0)
        diff = int(np.abs(full - np.asarray(saved["confusion_matrix"])).max())
        failures += int(n != saved["images"] or full.sum() != np.sum(saved["confusion_matrix"]))
        result = {
            "run": run_dir.name,
            "dataset": ds,
            "images": n,
            "checkpoint_step": step,
            "fracs": fracs,
            "zones": list(errors.ZONES),
            # [band width][zone] -> 12x12 confusion matrix, rows = ground truth
            "zone_confusion": total.tolist(),
            "check_vs_saved_metrics": {
                "metrics_file": str(metrics_path(run_dir, ds).relative_to(run_dir)),
                "same_matrix": diff == 0,
                "max_abs_diff": diff,
            },
            "device": str(device),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        tmp = out / f"{ds}.json.tmp"
        tmp.write_text(json.dumps(result) + "\n")
        tmp.replace(out / f"{ds}.json")
        print(
            f"{run_dir.name} {ds}: {n} images, max diff vs saved {diff}, {result['seconds']} s",
            flush=True,
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--wait", action="store_true", help="wait for the GPU lock")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    z = cfg["zones"]
    base = paths.DATA / z["out_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failures = 0
    try:
        with runs.gpu_lock(run="t10-error-zones", wait=args.wait):
            for r in cfg["runs"].values():
                run_dir = resolve_run(r["name"])
                failures += zone_run(run_dir, cfg["datasets"], z["fracs"], base / r["name"], device)
    except runs.GpuBusy as e:
        print(e, file=sys.stderr)
        return 2
    if failures:
        print(f"{failures} test sets do not cover the saved metrics' pixels", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
