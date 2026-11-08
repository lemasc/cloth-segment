"""Other people in LIP images: wrong 'person' pixels on background (M11, analysis only).

LIP labels one person per image, so any other visible person is background in
the ground truth. CIHP labels every person. A model trained on both may parse
the extra people and be scored wrong. This module measures that on one image:

- **false person**: ground truth is background, prediction is any other class
  (every shared class except background belongs to a person).
- **interior**: not in the T10 edge band (`errors.edge_band`, width as a share
  of the image diagonal), so it is not a boundary error.
- **separate**: a predicted-person component (8-connected) that touches no
  ground-truth person pixel. A strict test for "another person", but other
  people often touch the labelled one, so it misses most cases.
- **blob ignore**: interior false-person components of at least `share` of the
  image, plus the false-person pixels within the edge band width of them
  (Manhattan distance, `ndimage.binary_dilation`). Ignoring them gives an
  analysis score that bounds the effect from above: the rule cannot tell
  another person from an object or an unlabelled main person. It is not an
  official score; `evaluate.py` is not changed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from cloth_segment import errors
from cloth_segment.data.labels import IGNORE, NUM_CLASSES

BACKGROUND = 0


def false_person(pred: np.ndarray, gt: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """(all false-person pixels, the interior ones outside the edge band of `width`)."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} != gt {gt.shape}")
    fp = (gt == BACKGROUND) & (pred != BACKGROUND)
    return fp, fp & ~errors.edge_band(gt, width)


def separate_person(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Predicted-person pixels in components (8-connected) that touch no ground-truth person."""
    lab, n = ndimage.label(pred != BACKGROUND, structure=errors.EIGHT)
    touches = np.zeros(n + 1, dtype=bool)
    touches[np.unique(lab[(gt != BACKGROUND) & (gt != IGNORE)])] = True
    return (lab > 0) & ~touches[lab]


def blob_ignore(fp: np.ndarray, fp_int: np.ndarray, width: int, min_px: float) -> np.ndarray:
    """Interior false-person components of >= `min_px` pixels, grown by `width` within `fp`."""
    lab, n = ndimage.label(fp_int, structure=errors.EIGHT)
    if n == 0:
        return np.zeros(fp.shape, dtype=bool)
    sizes = np.bincount(lab.ravel())[1:]
    big = np.flatnonzero(sizes >= min_px) + 1
    if big.size == 0:
        return np.zeros(fp.shape, dtype=bool)
    return ndimage.binary_dilation(np.isin(lab, big), iterations=width) & fp


def image_result(
    pred: np.ndarray, gt: np.ndarray, frac: float, shares: Sequence[float]
) -> tuple[dict, np.ndarray]:
    """Per-image counts, and (1 + len(shares), C, C) confusion matrices.

    Matrix 0 scores every pixel (as `evaluate.py`); matrix 1 + i leaves out the
    pixels `blob_ignore` drops at `shares[i]` of the image area.
    """
    keep = gt != IGNORE
    if (gt[keep] >= NUM_CLASSES).any() or (pred >= NUM_CLASSES).any():
        raise ValueError(f"labels outside 0..{NUM_CLASSES - 1} and {IGNORE}")
    h, w = gt.shape
    width = errors.band_px(h, w, frac)
    fp, fp_int = false_person(pred, gt, width)
    sep = separate_person(pred, gt)
    person = keep & (gt != BACKGROUND)
    stats = {
        "h": h,
        "w": w,
        "bg": int((gt == BACKGROUND).sum()),
        "person_gt": int(person.sum()),
        "person_share": float(person.sum() / max(int(keep.sum()), 1)),
        "fp_all": int(fp.sum()),
        "fp_int": int(fp_int.sum()),
        "fp_int_sep": int((fp_int & sep).sum()),
    }
    cms = np.zeros((1 + len(shares), NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cms[0] = errors._cm(gt[keep], pred[keep])
    for i, s in enumerate(shares):
        drop = blob_ignore(fp, fp_int, width, s * h * w)
        sel = keep & ~drop
        cms[1 + i] = errors._cm(gt[sel], pred[sel])
        stats[f"ignored_{s}"] = int(drop.sum())
        stats[f"fp_int_ignored_{s}"] = int((drop & fp_int).sum())
    return stats, cms


def concentration(values: np.ndarray, tops: Sequence[float]) -> dict[str, float]:
    """Share of the total held by the largest `round(top * n)` values, per `top`."""
    v = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    total = v.sum()
    out = {}
    for t in tops:
        k = int(round(len(v) * t))
        out[str(t)] = float(v[:k].sum() / total) if total > 0 else 0.0
    return out
