"""Where the wrong pixels are: edge, swapped region or other interior (T10, analysis only).

Every scored pixel (ground truth not IGNORE) of a test image goes into one
zone, decided from the ground truth and the prediction of that image:

- **edge**: within `width` pixels (Chebyshev distance) of a ground-truth
  boundary, i.e. a pixel whose 4-neighbour has another label (IGNORE counts as
  a label here, so the edge of an ignored area is an edge too). `width` is a
  share of the image diagonal, so the band scales with the image.
- **swap**: not edge, and inside a ground-truth region (an 8-connected
  component of one class) whose most frequent prediction is another class:
  the model gave the whole region the wrong label.
- **inner**: everything else (not edge, region mostly right).

One confusion matrix per zone; the three add up to the image's confusion
matrix, so summed over a test set they add up to the saved `metrics.json`
matrix (same prediction path as `evaluate.py`, which is not changed). Wrong
pixels are the off-diagonal entries of each zone.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from scipy import ndimage

from cloth_segment.data.labels import IGNORE, NUM_CLASSES

ZONES: tuple[str, ...] = ("edge", "swap", "inner")
EIGHT = np.ones((3, 3), dtype=bool)


def band_px(h: int, w: int, frac: float) -> int:
    """Edge band width in pixels: `frac` of the image diagonal, at least 1."""
    if frac <= 0:
        raise ValueError(f"band fraction must be > 0, got {frac}")
    return max(1, round(frac * float(np.hypot(h, w))))


def boundary(gt: np.ndarray) -> np.ndarray:
    """Pixels whose right or lower 4-neighbour has another label, marked on both sides."""
    edge = np.zeros(gt.shape, dtype=bool)
    dx = gt[:, 1:] != gt[:, :-1]
    dy = gt[1:, :] != gt[:-1, :]
    edge[:, 1:] |= dx
    edge[:, :-1] |= dx
    edge[1:, :] |= dy
    edge[:-1, :] |= dy
    return edge


def edge_band(gt: np.ndarray, width: int) -> np.ndarray:
    """Pixels within Chebyshev distance `width - 1` of a boundary pixel (width 1 = boundary)."""
    edge = boundary(gt)
    if width <= 1 or not edge.any():
        return edge
    return ndimage.maximum_filter(edge, size=2 * width - 1)


def swapped_regions(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Pixels of ground-truth regions whose most frequent prediction is another class.

    A region is an 8-connected component of one class. Ties go to the lowest
    class id (`argmax`), which is the region's own class only if it is lowest.
    """
    out = np.zeros(gt.shape, dtype=bool)
    for c in np.unique(gt):
        if c == IGNORE:
            continue
        lab, n = ndimage.label(gt == c, structure=EIGHT)
        if n == 0:
            continue
        # counts[k, p] = pixels of component k+1 predicted as class p
        counts = np.zeros((n + 1, NUM_CLASSES), dtype=np.int64)
        np.add.at(counts, (lab.ravel(), pred.ravel()), 1)
        major = counts[1:].argmax(1)
        bad = np.flatnonzero(major != c) + 1
        if bad.size:
            out |= np.isin(lab, bad)
    return out


def _cm(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    idx = gt.astype(np.int64) * NUM_CLASSES + pred.astype(np.int64)
    return np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def zone_confusions(pred: np.ndarray, gt: np.ndarray, fracs: Sequence[float]) -> np.ndarray:
    """(len(fracs), 3, C, C) int64: per band width, one confusion matrix per zone.

    Rows = ground truth, columns = prediction; IGNORE pixels are skipped. Fails
    on any label outside the classes, like `evaluate.confusion_matrix`.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} != gt {gt.shape}")
    keep = gt != IGNORE
    if (gt[keep] >= NUM_CLASSES).any() or (pred >= NUM_CLASSES).any():
        raise ValueError(f"labels outside 0..{NUM_CLASSES - 1} and {IGNORE}")
    swap = swapped_regions(pred, gt)
    h, w = gt.shape
    out = np.zeros((len(fracs), len(ZONES), NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for i, f in enumerate(fracs):
        edge = edge_band(gt, band_px(h, w, f))
        zones = (edge, ~edge & swap, ~edge & ~swap)
        for z, m in enumerate(zones):
            sel = m & keep
            out[i, z] = _cm(gt[sel], pred[sel])
    return out


def error_shares(zcm: np.ndarray) -> dict[str, float | None]:
    """Share of wrong pixels in each zone, from one (3, C, C) zone stack."""
    off = [int(m.sum() - np.trace(m)) for m in zcm]
    total = sum(off)
    return {z: (o / total if total else None) for z, o in zip(ZONES, off, strict=True)}


def class_error_shares(zcm: np.ndarray) -> np.ndarray:
    """(C, 3): per ground-truth class, the share of its wrong pixels in each zone (NaN if none)."""
    wrong = np.stack([m.sum(1) - np.diag(m) for m in zcm], axis=1).astype(np.float64)
    tot = wrong.sum(1, keepdims=True)
    return np.divide(wrong, tot, out=np.full(wrong.shape, np.nan), where=tot > 0)


def iter_predictions(model, loader, device, dtype) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (dataset id, pred, ground truth) per image, numpy (H, W) at original resolution.

    Same forward path as `evaluate.predict_confusions` (eval transform, restore
    to the original size, argmax on float32 logits).
    """
    import torch

    from cloth_segment.evaluate import autocast, restore_logits
    from cloth_segment.models import logits_at

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            with autocast(device, dtype):
                logits = model(x)
            logits = logits_at(logits.float(), x.shape[-2:])
            for k in range(x.shape[0]):
                full = restore_logits(logits[k], batch["valid_hw"][k], batch["orig_hw"][k])
                pred = full.argmax(0).to(torch.uint8).cpu().numpy()
                yield int(batch["dataset"][k]), pred, batch["orig_mask"][k].numpy()
