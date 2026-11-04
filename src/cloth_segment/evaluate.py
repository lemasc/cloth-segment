"""Evaluation protocol and metrics (shared contract, CONTRIBUTING section 4).

Entry point: `python -m cloth_segment.evaluate <run> [test_set ...]` (T5).
Scores the run's `ckpt/best.pt` on each test set (default: lip, cihp, dfmm,
whatever the run trained on, so cross-dataset tables need no extra step) and
writes `eval/<EVAL_VERSION>/<test_set>/metrics.json` into the run folder. An
existing file is never overwritten: that is the cache. Reports are T9.

Protocol (eval-v1):
- Predictions are scored at the **original mask resolution**. Logits from the
  eval transform (long side -> size, pad bottom/right) are cropped to
  `valid_hw`, resized bilinearly to `orig_hw`, then argmaxed.
- One confusion matrix per test set, summed over all its pixels
  (dataset-level, not a mean of per-image scores). IGNORE pixels are skipped.
- Per-class IoU = TP / (TP + FP + FN), Dice = 2TP / (2TP + FP + FN).
- mIoU and mean Dice average over the classes that have ground-truth pixels in
  that test set, background included. A class with no ground truth is not in
  the mean (its IoU is 0 or undefined there). Pixels wrongly predicted as it
  are still false negatives of their true class, so they lower that class's
  IoU, and they show in the confusion matrix.
- No test-time augmentation. Autocast follows the run's `train.amp`; the
  argmax is taken on float32 logits.
- The checkpoint is `best.pt` (highest mIoU on the internal val split), never
  one chosen on a test set (D4).

Changing any of this changes reported numbers: bump EVAL_VERSION.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cloth_segment import paths, runs
from cloth_segment.data.dataset import DATASET_NAMES, SegDataset, collate, load_split, read_mask
from cloth_segment.data.labels import IGNORE, NUM_CLASSES, SHARED_CLASSES
from cloth_segment.models import build_model, logits_at

EVAL_VERSION = "eval-v1"


def restore_logits(
    logits: torch.Tensor, valid_hw: Sequence[int], orig_hw: Sequence[int]
) -> torch.Tensor:
    """(C, S, S) eval-transform logits -> (C, H, W) logits at the original resolution."""
    if logits.ndim != 3:
        raise ValueError(f"expected (C, S, S) logits, got {tuple(logits.shape)}")
    vh, vw = int(valid_hw[0]), int(valid_hw[1])
    if vh > logits.shape[1] or vw > logits.shape[2]:
        raise ValueError(f"valid_hw {valid_hw} larger than logits {tuple(logits.shape)}")
    crop = logits[None, :, :vh, :vw].float()
    out = F.interpolate(crop, size=tuple(int(s) for s in orig_hw), mode="bilinear")
    return out[0]


def confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES
) -> torch.Tensor:
    """(C, C) int64 counts, rows = ground truth, columns = prediction. Skips IGNORE.

    Raises on any target value that is neither a class ID nor IGNORE, and on
    any prediction outside the class IDs, so a wrong mask cannot be scored.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    pred, target = pred.reshape(-1).long(), target.reshape(-1).long()
    keep = target != IGNORE
    pred, target = pred[keep], target[keep]
    if target.numel() and (target.min() < 0 or target.max() >= num_classes):
        raise ValueError(f"target has values outside 0..{num_classes - 1} and {IGNORE}")
    if pred.numel() and (pred.min() < 0 or pred.max() >= num_classes):
        raise ValueError(f"pred has values outside 0..{num_classes - 1}")
    idx = target * num_classes + pred
    counts = torch.bincount(idx, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).cpu()


def scores(cm: torch.Tensor, class_names: Sequence[str] = SHARED_CLASSES) -> dict:
    """Per-class IoU / Dice and their means from a confusion matrix (see module doc).

    Per-class values are None where the class has neither ground truth nor
    predictions (0/0). `in_mean` lists the classes averaged into mIoU / mDice.
    """
    cm = cm.double()
    if cm.shape != (len(class_names), len(class_names)):
        raise ValueError(f"confusion matrix {tuple(cm.shape)} vs {len(class_names)} classes")
    tp = cm.diag()
    gt = cm.sum(dim=1)
    pr = cm.sum(dim=0)
    fp, fn = pr - tp, gt - tp
    union = tp + fp + fn
    iou = torch.where(union > 0, tp / union.clamp(min=1), torch.nan)
    dice = torch.where(union > 0, 2 * tp / (2 * tp + fp + fn).clamp(min=1), torch.nan)
    present = gt > 0
    total = cm.sum()

    def _list(t: torch.Tensor) -> dict[str, float | None]:
        return {n: (None if torch.isnan(v) else float(v)) for n, v in zip(class_names, t)}

    return {
        "eval_version": EVAL_VERSION,
        "miou": float(iou[present].mean()) if present.any() else None,
        "mdice": float(dice[present].mean()) if present.any() else None,
        "pixel_acc": float(tp.sum() / total) if total > 0 else None,
        "iou": _list(iou),
        "dice": _list(dice),
        "gt_pixels": {n: int(v) for n, v in zip(class_names, gt)},
        "in_mean": [n for n, p in zip(class_names, present) if p],
    }


AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}


def amp_dtype(name: str, device: torch.device) -> torch.dtype | None:
    """Autocast dtype for `train.amp` on this device; None = full precision."""
    if name not in AMP_DTYPES:
        raise ValueError(f"train.amp must be one of {sorted(AMP_DTYPES)}, got {name!r}")
    if device.type != "cuda":
        return None
    dtype = AMP_DTYPES[name]
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("train.amp is bf16 but this GPU does not support bf16")
    return dtype


def autocast(device: torch.device, dtype: torch.dtype | None):
    """Autocast context for `dtype`, or a no-op for full precision."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=dtype)


class WithOriginalMask(Dataset):
    """Eval samples plus `orig_mask`, the mask at its original resolution (eval-v1).

    Read in the loader workers, so scoring does not wait on PNG decoding.
    """

    def __init__(self, dataset: SegDataset):
        if dataset.train:
            raise ValueError("evaluation needs a dataset built with train=False")
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> dict:
        sample = self.dataset[i]
        sample["orig_mask"] = read_mask(self.dataset.mask_root / self.dataset.rows.at[i, "mask"])
        return sample


def eval_collate(batch: list[dict]) -> dict:
    """`collate`, keeping the differently sized original masks as a list."""
    masks = [b.pop("orig_mask") for b in batch]
    out = collate(batch)
    out["orig_mask"] = masks
    return out


def eval_loader(dataset: SegDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        WithOriginalMask(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=eval_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


@torch.no_grad()
def predict_confusion(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run `model` over an `eval_loader` and sum one confusion matrix (eval-v1).

    Leaves the model in eval mode; the caller switches it back to train.
    """
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with autocast(device, dtype):
            logits = model(x)
        logits = logits_at(logits.float(), x.shape[-2:])
        for k in range(x.shape[0]):
            full = restore_logits(logits[k], batch["valid_hw"][k], batch["orig_hw"][k])
            target = batch["orig_mask"][k].to(device, non_blocking=True)
            cm += confusion_matrix(full.argmax(0), target)
    return cm


# ---------------------------------------------------------------- test-set entry point

TEST_SETS: tuple[str, ...] = DATASET_NAMES  # test split of each dataset (D4)
CHECKPOINT = "best.pt"


def resolve_run(run: str | Path, runs_dir: Path = paths.RUNS) -> Path:
    """A run folder given as a path or as a name under `runs_dir`; must be finished."""
    run_dir = Path(run)
    if not (run_dir / "meta.json").exists():
        run_dir = runs_dir / str(run)
    if not (run_dir / "meta.json").exists():
        raise FileNotFoundError(f"no run folder {run!r} (looked in {runs_dir})")
    status = runs.read_meta(run_dir).get("status")
    if status != "finished":
        raise ValueError(f"{run_dir.name}: status is {status!r}, not finished")
    return run_dir


def metrics_path(run_dir: Path, test_set: str) -> Path:
    return run_dir / "eval" / EVAL_VERSION / test_set / "metrics.json"


def load_test_set(cfg: dict, test_set: str) -> SegDataset:
    """The test split of one dataset, with the run's data version and input size."""
    if test_set not in TEST_SETS:
        raise ValueError(f"test set must be one of {TEST_SETS}, got {test_set!r}")
    one = cfg | {"data": cfg["data"] | {"datasets": [test_set]}}
    return load_split(one, "test", train=False)


def load_model(run_dir: Path, cfg: dict, device: torch.device) -> tuple[nn.Module, int]:
    """The run's model with `ckpt/best.pt` loaded; returns (model, checkpoint step)."""
    state = torch.load(run_dir / "ckpt" / CHECKPOINT, map_location="cpu", weights_only=True)
    model = build_model(cfg["model"])
    model.load_state_dict(state["model"])
    return model.to(device).eval(), int(state["step"])


def evaluate_run(
    run_dir: Path,
    test_sets: Sequence[str],
    device: torch.device,
    *,
    num_workers: int | None = None,
    make_dataset: Callable[[dict, str], SegDataset] = load_test_set,
) -> dict[str, Path]:
    """Score `best.pt` on each test set not scored yet; returns test set -> metrics.json."""
    meta = runs.read_meta(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())  # resolved at run start
    todo = [t for t in test_sets if not metrics_path(run_dir, t).exists()]
    out = {t: metrics_path(run_dir, t) for t in test_sets}
    if not todo:
        return out
    model, step = load_model(run_dir, cfg, device)
    dtype = amp_dtype(cfg["train"]["amp"], device)
    workers = (cfg.get("run") or {}).get("num_workers", 0) if num_workers is None else num_workers
    for test_set in todo:
        t0 = time.perf_counter()
        dataset = make_dataset(cfg, test_set)
        loader = eval_loader(dataset, cfg["train"]["batch_size"], workers)
        cm = predict_confusion(model, loader, device, dtype)
        result = {
            "run": run_dir.name,
            "config_hash": meta["config_hash"],
            "data_version": cfg["data"]["version"],
            "test_set": test_set,
            "split": "test",
            "images": len(dataset),
            "checkpoint": CHECKPOINT,
            "checkpoint_step": step,
            **scores(cm, SHARED_CLASSES),
            "confusion_matrix": cm.tolist(),  # rows = ground truth, columns = prediction
            "classes": list(SHARED_CLASSES),
            "eval_commit": "{}{}".format(*_commit()),
            "device": str(device),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        path = out[test_set]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(path)
        print(f"{test_set}: mIoU {result['miou']:.4f} ({len(dataset)} images) -> {path}")
    return out


def _commit() -> tuple[str, str]:
    try:
        commit, dirty = runs.git_state()
    except Exception:  # not a git checkout: still write metrics, mark the commit unknown
        return "unknown", ""
    return commit, "-dirty" if dirty else ""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score a finished run on the test sets (eval-v1).")
    p.add_argument("run", help="run folder, or its name under runs/")
    p.add_argument("test_sets", nargs="*", default=list(TEST_SETS), help=f"{TEST_SETS}")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=None, help="default: the run's num_workers")
    p.add_argument("--wait", action="store_true", help="wait for the GPU lock instead of failing")
    args = p.parse_args(argv)
    bad = [t for t in args.test_sets if t not in TEST_SETS]
    if bad:
        p.error(f"unknown test sets {bad}; choose from {TEST_SETS}")
    run_dir = resolve_run(args.run)
    device = torch.device(args.device)
    lock = (
        runs.gpu_lock(f"eval {run_dir.name}", wait=args.wait)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    try:
        with lock:
            evaluate_run(run_dir, args.test_sets, device, num_workers=args.workers)
    except runs.GpuBusy as e:
        print(f"{e}; use --wait to queue", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
