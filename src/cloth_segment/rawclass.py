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
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cloth_segment.data.dataset import SegDataset, collate, read_mask
from cloth_segment.data.labels import IGNORE, MAPPING, NUM_CLASSES, SHARED_CLASSES, convert
from cloth_segment.data.sources import RAW_LABELS, list_pairs

ACCESSORY = SHARED_CLASSES.index("accessory")
KEY_CLASSES = ("accessory", "headwear", "face", "hair", "background")


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


RAW_MAX = 256  # raw masks are 8-bit


def raw_pred_matrix(pred: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """(RAW_MAX, NUM_CLASSES) int64 counts: row = raw id, column = predicted shared class.

    Covers every pixel, including raw classes that map to IGNORE.
    """
    if pred.shape != raw.shape:
        raise ValueError(f"shapes differ: {tuple(pred.shape)} {tuple(raw.shape)}")
    idx = raw.long().flatten() * NUM_CLASSES + pred.long().flatten()
    return torch.bincount(idx, minlength=RAW_MAX * NUM_CLASSES).view(RAW_MAX, NUM_CLASSES)


def shared_counts(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(3, NUM_CLASSES) int64: ground-truth, predicted and true-positive pixels per class.

    IGNORE pixels are skipped, as in `evaluate.confusion_matrix`, so the sums
    over a test set equal the row sums, column sums and diagonal of its
    saved confusion matrix.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shapes differ: {tuple(pred.shape)} {tuple(target.shape)}")
    keep = target != IGNORE
    t, p = target[keep].long(), pred[keep].long()
    return torch.stack(
        [
            torch.bincount(t, minlength=NUM_CLASSES),
            torch.bincount(p, minlength=NUM_CLASSES),
            torch.bincount(t[t == p], minlength=NUM_CLASSES),
        ]
    )


def raw_rows(matrix: torch.Tensor, dataset: str) -> list[dict]:
    """One dict per raw class present in a `raw_pred_matrix` (one image or summed).

    `hit_px` counts its pixels predicted as the shared class it maps to (None
    for raw classes that map to IGNORE).
    """
    names, mapping = RAW_LABELS[dataset], MAPPING[dataset]
    m = matrix.cpu()
    out = []
    for rid in torch.nonzero(m.sum(1)).flatten().tolist():
        if rid not in names:
            raise ValueError(f"{dataset}: raw id {rid} is not a known raw class")
        sid = mapping.get(rid, IGNORE)
        out.append(
            {
                "raw_id": rid,
                "raw_name": names[rid],
                "shared": SHARED_CLASSES[sid] if sid != IGNORE else None,
                "raw_px": int(m[rid].sum()),
                "hit_px": int(m[rid, sid]) if sid != IGNORE else None,
            }
        )
    return out


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


@torch.no_grad()
def iter_predictions(model, loader: DataLoader, device, dtype):
    """Yield (row index, pred, shared target, raw mask) per image, each (H, W) on `device`.

    Same forward path as `tally_loader`; for per-image analysis and figures.
    """
    from cloth_segment.evaluate import autocast, restore_logits
    from cloth_segment.models import logits_at

    model.eval()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with autocast(device, dtype):
            logits = model(x)
        logits = logits_at(logits.float(), x.shape[-2:])
        for k in range(x.shape[0]):
            full = restore_logits(logits[k], batch["valid_hw"][k], batch["orig_hw"][k])
            yield (
                int(batch["index"][k]),
                full.argmax(0),
                batch["orig_mask"][k].to(device),
                batch["raw_mask"][k].to(device),
            )


def box_stats(raw: np.ndarray, rid: int, around: int) -> dict | None:
    """Pixels of raw class `rid` in one raw mask, its bounding box, how much of
    the box it fills, and the share of the box that is raw class `around`.
    None if `rid` is absent."""
    sel = raw == rid
    if not sel.any():
        return None
    ys, xs = np.nonzero(sel)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    box = raw[y0:y1, x0:x1]
    n = int(sel.sum())
    return {
        "pixels": n,
        "box": (y0, y1, x0, x1),
        "box_w": x1 - x0,
        "box_h": y1 - y0,
        "fill": n / box.size,
        "around_share": int((box == around).sum()) / box.size,
        "image_hw": raw.shape[:2],
    }


def accessory_supply(manifest: pd.DataFrame) -> pd.DataFrame:
    """`accessory` images and pixels per dataset and split, from the splits manifest.

    Excluded rows are dropped. "% of labelled pixels" counts shared classes only
    (IGNORE pixels left out), the same pixels the metrics score.
    """
    cols = [f"c{i}" for i in range(NUM_CLASSES)]
    rows = manifest[manifest["split"] != "excluded"]
    out = []
    for (ds, split), g in rows.groupby(["dataset", "split"], sort=False):
        acc = g[f"c{ACCESSORY}"]
        labelled = int(g[cols].to_numpy().sum())
        out.append(
            {
                "dataset": ds,
                "split": split,
                "images": len(g),
                "images_with_accessory": int((acc > 0).sum()),
                "pct_images": 100 * float((acc > 0).mean()) if len(g) else 0.0,
                "accessory_pixels": int(acc.sum()),
                "pct_labelled_pixels": 100 * int(acc.sum()) / labelled if labelled else 0.0,
            }
        )
    return pd.DataFrame(out)


def miou_without(metrics: dict, drop: str = "accessory") -> dict:
    """12-class mIoU, mIoU without `drop`, and `drop`'s IoU, from one `metrics.json`.

    Same rule as `evaluate`: a class counts in the mean when it has ground-truth
    pixels (row sum of the confusion matrix > 0).
    """
    names = list(metrics["iou"])
    if drop not in names:
        raise ValueError(f"{drop!r} is not a class in this metrics file")
    present = [n for n, row in zip(names, metrics["confusion_matrix"], strict=True) if sum(row)]
    kept = [metrics["iou"][n] for n in present if n != drop]
    return {
        "miou_all": metrics["miou"],
        f"miou_without_{drop}": sum(kept) / len(kept) if kept else None,
        drop: metrics["iou"][drop],
    }


def headwear_summary(df: pd.DataFrame) -> dict:
    """Pixel-weighted summary of one run's per-image sunglasses predictions.

    `df` has `sunglasses_px`, `hat_px_gt` and one `pred_<class>` share column
    per shared class (shares of that image's sunglasses pixels). Shares are
    weighted by sunglasses pixels, so they are pixel ratios over all images.
    """

    def share(sub: pd.DataFrame, col: str) -> float | None:
        w = sub["sunglasses_px"].sum()
        return float((sub[col] * sub["sunglasses_px"]).sum() / w) if w else None

    hat = df["hat_px_gt"] > 0
    head = df["pred_headwear"]
    return {
        "images": len(df),
        "images_with_gt_hat": int(hat.sum()),
        "pixel_share_all": {
            c: share(df, f"pred_{c}")
            for c in SHARED_CLASSES
            if (share(df, f"pred_{c}") or 0) > 0.005
        },
        "by_gt_hat": {
            tag: {
                "images": len(sub),
                "sunglasses_px": int(sub["sunglasses_px"].sum()),
                **{c: share(sub, f"pred_{c}") for c in KEY_CLASSES},
            }
            for tag, sub in (("with_gt_hat", df[hat]), ("without_gt_hat", df[~hat]))
        },
        "images_over_half_headwear": int((head > 0.5).sum()),
        "images_over_half_headwear_with_gt_hat": int(((head > 0.5) & hat).sum()),
    }
