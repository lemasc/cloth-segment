"""Evaluation protocol and metrics (shared contract, CONTRIBUTING section 4).

Entry point: `python -m cloth_segment.evaluate <run> <test_set>`, cached per
(run, test set, EVAL_VERSION); writes `eval/<test_set>/metrics.json`. The
metric code is T5; the run loading and reports are T5/T9.

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
- No test-time augmentation.

Changing any of this changes reported numbers: bump EVAL_VERSION.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cloth_segment.data.dataset import SegDataset, collate, read_mask
from cloth_segment.data.labels import IGNORE, NUM_CLASSES, SHARED_CLASSES
from cloth_segment.models import logits_at

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


def main() -> None:
    raise NotImplementedError("T5/T9: test-set entry point (run loading, metrics.json)")


if __name__ == "__main__":
    main()
