"""Score the shared `accessory` class per raw class (M6, analysis only).

`accessory` merges several raw classes (LIP/CIHP glove, scarf, sunglasses; DFMM
eyeglass, neckwear, ...). The model predicts the merged class, so a raw class
has no precision of its own. For each raw class this reports:

- `recall`: its pixels predicted as `accessory` / its pixels;
- `where`: how its pixels are spread over the predicted shared classes;
- `iou_others_ignored`: TP / (raw pixels + false positives), where a false
  positive is an `accessory` prediction on a pixel that is neither `accessory`
  nor IGNORE. Pixels of the other raw accessory classes are left out, so the
  value does not punish a raw class for predictions that hit a sibling class;
- `image_hit_rate`: share of images containing it where at least half of its
  pixels are predicted `accessory`.

Uses the eval pipeline of `evaluate.py` unchanged (same transform, same
restore to the original resolution), so predictions match the saved metrics.
Nothing here changes the eval protocol or any label.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cloth_segment.data.dataset import SegDataset, collate, read_mask
from cloth_segment.data.labels import IGNORE, MAPPING, NUM_CLASSES, SHARED_CLASSES, convert
from cloth_segment.data.sources import RAW_LABELS, list_pairs

ACCESSORY = SHARED_CLASSES.index("accessory")


def accessory_raw_ids(dataset: str) -> dict[int, str]:
    """Raw id -> raw name of every raw class of `dataset` that maps to `accessory`."""
    return {rid: RAW_LABELS[dataset][rid] for rid, s in MAPPING[dataset].items() if s == ACCESSORY}


def image_counts(
    pred: torch.Tensor, target: torch.Tensor, raw: torch.Tensor, raw_ids: Sequence[int]
) -> tuple[dict[int, torch.Tensor], int]:
    """One image. `pred`, `target`, `raw` are (H, W).

    Returns {raw id: (NUM_CLASSES,) counts of predicted shared class on that raw
    class's pixels} and the count of false-positive `accessory` pixels.
    """
    if not (pred.shape == target.shape == raw.shape):
        raise ValueError(
            f"shapes differ: {tuple(pred.shape)} {tuple(target.shape)} {tuple(raw.shape)}"
        )
    per_raw = {}
    for rid in raw_ids:
        sel = raw == rid
        per_raw[rid] = torch.bincount(pred[sel].long(), minlength=NUM_CLASSES).cpu()
    fp = int(((pred == ACCESSORY) & (target != ACCESSORY) & (target != IGNORE)).sum())
    return per_raw, fp


class RawClassTally:
    """Adds `image_counts` over many images of one dataset."""

    def __init__(self, raw_ids: dict[int, str]):
        self.names = dict(raw_ids)
        self.dist = {r: torch.zeros(NUM_CLASSES, dtype=torch.int64) for r in raw_ids}
        self.images = dict.fromkeys(raw_ids, 0)
        self.hits = dict.fromkeys(raw_ids, 0)
        self.fp = 0

    def add(self, per_raw: dict[int, torch.Tensor], fp: int) -> None:
        self.fp += fp
        for rid, d in per_raw.items():
            n = int(d.sum())
            if n == 0:
                continue
            self.dist[rid] += d
            self.images[rid] += 1
            self.hits[rid] += int(2 * int(d[ACCESSORY]) >= n)

    def result(self) -> dict:
        out = {}
        for rid, name in self.names.items():
            d = self.dist[rid]
            gt, tp = int(d.sum()), int(d[ACCESSORY])
            out[name] = {
                "raw_id": rid,
                "gt_pixels": gt,
                "images": self.images[rid],
                "tp_pixels": tp,
                "recall": tp / gt if gt else None,
                "iou_others_ignored": tp / (gt + self.fp) if gt else None,
                "image_hit_rate": self.hits[rid] / self.images[rid] if self.images[rid] else None,
                "where": {c: int(d[i]) for i, c in enumerate(SHARED_CLASSES) if int(d[i])},
            }
        return {"false_positive_pixels": self.fp, "raw_classes": out}


def _read_raw(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode not in ("L", "P"):
            raise ValueError(f"{path}: raw mask mode {im.mode}, expected L or P")
        return np.asarray(im).copy()


class WithRawMask(Dataset):
    """Eval samples plus `orig_mask` (shared) and `raw_mask` (raw ids), original resolution.

    Fails if the raw mask does not convert to the stored shared mask, so a
    wrong pairing cannot be scored.
    """

    def __init__(self, dataset: SegDataset):
        if dataset.train:
            raise ValueError("evaluation needs a dataset built with train=False")
        self.dataset = dataset
        rows = dataset.rows
        if rows.dataset.nunique() != 1:
            raise ValueError("one dataset at a time")
        self.name = str(rows.dataset.iloc[0])
        lookup = {}
        for split in rows.source_split.unique():
            lookup |= {(split, p.id): p.mask for p in list_pairs(self.name, split)}
        self.raw_paths = [lookup[(s, i)] for s, i in zip(rows.source_split, rows.id, strict=True)]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> dict:
        sample = self.dataset[i]
        shared = read_mask(self.dataset.mask_root / self.dataset.rows.at[i, "mask"])
        raw = _read_raw(self.raw_paths[i])
        if not np.array_equal(convert(raw, self.name), shared.numpy()):
            raise ValueError(f"{self.raw_paths[i]}: raw mask does not convert to the shared mask")
        sample["orig_mask"] = shared
        sample["raw_mask"] = torch.from_numpy(raw)
        return sample


def raw_collate(batch: list[dict]) -> dict:
    keep = {k: [b.pop(k) for b in batch] for k in ("orig_mask", "raw_mask")}
    return collate(batch) | keep


def raw_loader(dataset: SegDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        WithRawMask(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=raw_collate,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def tally_loader(model, loader: DataLoader, device, dtype, dataset: str) -> dict:
    """Run `model` over a `raw_loader` and return `RawClassTally.result()`."""
    from cloth_segment.evaluate import autocast, restore_logits
    from cloth_segment.models import logits_at

    tally = RawClassTally(accessory_raw_ids(dataset))
    model.eval()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with autocast(device, dtype):
            logits = model(x)
        logits = logits_at(logits.float(), x.shape[-2:])
        for k in range(x.shape[0]):
            full = restore_logits(logits[k], batch["valid_hw"][k], batch["orig_hw"][k])
            per_raw, fp = image_counts(
                full.argmax(0),
                batch["orig_mask"][k].to(device),
                batch["raw_mask"][k].to(device),
                list(tally.names),
            )
            tally.add(per_raw, fp)
    return tally.result()
