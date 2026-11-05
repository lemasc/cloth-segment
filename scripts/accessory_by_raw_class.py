"""Score `accessory` per raw class for finished runs (M6). Needs the GPU lock.

Writes one JSON with, per run and test set, recall / IoU / image hit rate for
each raw class that maps to `accessory` (see cloth_segment/rawclass.py).
Read-only on runs/ and data/. Cross-checks its accessory totals against the
run's saved eval metrics.

Usage: uv run python scripts/accessory_by_raw_class.py RUN [RUN ...] --out FILE.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch
import yaml

from cloth_segment import runs
from cloth_segment.evaluate import (
    amp_dtype,
    load_eval_set,
    load_model,
    resolve_run,
)
from cloth_segment.rawclass import raw_loader, tally_loader


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--datasets", nargs="+", default=["lip", "cihp", "dfmm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait", action="store_true", help="wait for the GPU lock")
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {}
    try:
        with runs.gpu_lock(run="accessory-by-raw-class", wait=args.wait):
            for name in args.runs:
                run_dir = resolve_run(name)
                cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
                model, step = load_model(run_dir, cfg, device)
                dtype = amp_dtype(cfg["train"]["amp"], device)
                workers = (cfg.get("run") or {}).get("num_workers", 0)
                result[run_dir.name] = {"checkpoint_step": step, "datasets": {}}
                for ds in args.datasets:
                    t0 = time.perf_counter()
                    data = load_eval_set(cfg, ds, "test")
                    res = tally_loader(
                        model,
                        raw_loader(data, cfg["train"]["batch_size"], workers),
                        device,
                        dtype,
                        ds,
                    )
                    res["images"] = len(data)
                    # cross-check against the saved shared-class confusion matrix
                    saved = next(iter(run_dir.glob(f"eval/*/{ds}/metrics.json")), None)
                    if saved is not None:
                        cm = json.loads(saved.read_text())["confusion_matrix"]
                        acc = [r for r in res["raw_classes"].values()]
                        res["check_vs_saved_metrics"] = {
                            "metrics_file": str(saved.relative_to(run_dir)),
                            "saved_gt": sum(cm[11]),
                            "here_gt": sum(r["gt_pixels"] for r in acc),
                            "saved_tp": cm[11][11],
                            "here_tp": sum(r["tp_pixels"] for r in acc),
                        }
                    res["seconds"] = round(time.perf_counter() - t0, 1)
                    result[run_dir.name]["datasets"][ds] = res
                    print(f"{run_dir.name} {ds}: done in {res['seconds']}s", flush=True)
    except runs.GpuBusy as e:
        print(e, file=sys.stderr)
        return 2
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
