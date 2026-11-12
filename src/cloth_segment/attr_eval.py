"""Attribute evaluation for multi-task runs. Separate from `evaluate.py`, which scores segmentation.

Entry point: `python -m cloth_segment.attr_eval <run> [--wait]`. Scores the
run's `ckpt/best.pt` attribute heads on the DFMM test split and writes
`eval/<ATTR_EVAL_VERSION>/dfmm/attr_metrics.json` into the run folder (never
overwritten: the cache). Segmentation of the same run is scored by
`evaluate.py` as for every other model.

Protocol (attr-v1):
- Same images and eval transform as `evaluate.py` (one prediction per image,
  no test-time augmentation; autocast follows `train.amp`).
- Targets are the loader's codes with NA, missing and `model.attr.ignore_values`
  set to IGNORE; those items are left out of that head.
- Per head: accuracy and macro-F1. The headline macro-F1 averages the classes
  with at least `MIN_COUNT` true test items; rarer classes are reported with
  their F1 but not averaged (too few to score). `macro_f1_present` averages
  every class present.
- Baseline: the most frequent labelled train value of each head, predicted for
  every test image, scored the same way.
- The run's headline number is the mean over heads of the headline macro-F1.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from cloth_segment import attributes, evaluate, runs
from cloth_segment.data.dataset import ATTR_NAMES, attribute_targets, select_rows

ATTR_EVAL_VERSION = "attr-v1"
MIN_COUNT = 20  # true test items a class needs to count in the headline macro-F1
DATASET = "dfmm"  # the only dataset with attribute labels


def predict_attrs(
    model: nn.Module, loader: DataLoader, device: torch.device, dtype: torch.dtype | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(targets, predictions), both (N, heads), over an eval loader. Leaves the model in eval."""
    spec = model.spec
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            with evaluate.autocast(device, dtype):
                _, logits = model.forward_multi(x)
            ps.append(torch.stack([lg.float().argmax(1) for lg in logits], dim=1).cpu())
            ys.append(attributes.head_targets(batch["attrs"], spec))
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def train_targets(cfg: dict, spec: attributes.AttrSpec) -> np.ndarray:
    """(N, heads) targets of the DFMM train split, for the majority baseline."""
    from cloth_segment.data.splits import load_manifest

    version = cfg["data"]["version"]
    df, _ = load_manifest(version.rsplit("-", 1)[0])
    rows = select_rows(df, "train", [DATASET])
    codes = torch.from_numpy(attribute_targets(rows[list(ATTR_NAMES)].to_numpy()))
    return attributes.head_targets(codes, spec).numpy()


def metrics_path(run_dir: Path) -> Path:
    return run_dir / "eval" / ATTR_EVAL_VERSION / DATASET / "attr_metrics.json"


def evaluate_attr_run(
    run_dir: Path, device: torch.device, *, num_workers: int | None = None
) -> Path:
    """Score `best.pt`'s attribute heads on the DFMM test split; returns the metrics file."""
    out = metrics_path(run_dir)
    if out.exists():
        return out
    meta = runs.read_meta(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model, step = evaluate.load_model(run_dir, cfg, device)
    if not hasattr(model, "forward_multi"):
        raise ValueError(f"{run_dir.name}: model {cfg['model']['name']!r} has no attribute heads")
    spec = model.spec
    t0 = time.perf_counter()
    dataset = evaluate.load_eval_set(cfg, DATASET, "test")
    workers = (cfg.get("run") or {}).get("num_workers", 0) if num_workers is None else num_workers
    loader = evaluate.eval_loader(dataset, cfg["train"]["batch_size"], workers)
    y, pred = predict_attrs(model, loader, device, evaluate.amp_dtype(cfg["train"]["amp"], device))
    base = attributes.majority_predictions(train_targets(cfg, spec), len(y), spec)
    result = {
        "run": run_dir.name,
        "config_hash": meta["config_hash"],
        "data_version": cfg["data"]["version"],
        "attr_eval_version": ATTR_EVAL_VERSION,
        "dataset": DATASET,
        "split": "test",
        "images": len(dataset),
        "checkpoint": evaluate.CHECKPOINT,
        "checkpoint_step": step,
        "checkpoint_select": cfg["eval"]["select"],
        "min_count": MIN_COUNT,
        "heads": list(spec.heads),
        "ignore_values": {h: [spec.value_names(h)[c] for c in cs] for h, cs in spec.ignore.items()},
        "model": attributes.all_metrics(y, pred, spec, MIN_COUNT),
        "majority_baseline": attributes.all_metrics(y, base, spec, MIN_COUNT),
        "eval_commit": "{}{}".format(*evaluate._commit()),
        "device": str(device),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(out)
    m, b = result["model"], result["majority_baseline"]
    print(
        f"attr {DATASET} test: mean macro-F1 {m['mean_macro_f1']:.4f} "
        f"(majority {b['mean_macro_f1']:.4f}), {len(dataset)} images -> {out}"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=f"Score a multi-task run's attributes ({ATTR_EVAL_VERSION})."
    )
    p.add_argument("run", help="run folder, or its name under runs/")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=None, help="default: the run's num_workers")
    p.add_argument("--wait", action="store_true", help="wait for the GPU lock instead of failing")
    args = p.parse_args(argv)
    run_dir = evaluate.resolve_run(args.run)
    device = torch.device(args.device)
    lock = (
        runs.gpu_lock(f"attr-eval {run_dir.name}", wait=args.wait)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    try:
        with lock:
            evaluate_attr_run(run_dir, device, num_workers=args.workers)
    except runs.GpuBusy as e:
        print(f"{e}; use --wait to queue", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
