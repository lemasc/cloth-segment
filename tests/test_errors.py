"""Error zones (edge / swapped region / inner): tiny synthetic masks, no data, no GPU."""

import numpy as np
import pytest

from cloth_segment import errors
from cloth_segment.data.labels import IGNORE, NUM_CLASSES, SHARED_CLASSES

BG, UPPER, OUTER, PANTS = (
    SHARED_CLASSES.index(c) for c in ("background", "upper", "outer", "pants")
)


def _scene() -> np.ndarray:
    """20x20 background with an upper block (rows 2-9) above a pants block (rows 10-17)."""
    gt = np.full((20, 20), BG, dtype=np.uint8)
    gt[2:10, 4:16] = UPPER
    gt[10:18, 4:16] = PANTS
    return gt


def test_band_px_scales_with_the_diagonal():
    assert errors.band_px(300, 400, 0.01) == 5
    assert errors.band_px(10, 10, 0.001) == 1
    with pytest.raises(ValueError):
        errors.band_px(10, 10, 0)


def test_boundary_marks_both_sides_and_band_grows():
    gt = np.zeros((5, 6), dtype=np.uint8)
    gt[:, 3:] = UPPER
    b = errors.boundary(gt)
    assert b[:, 2].all() and b[:, 3].all() and b.sum() == 10
    band = errors.edge_band(gt, 2)
    assert band[:, 1:5].all() and not band[:, 0].any() and not band[:, 5].any()


def test_whole_region_relabelled_is_a_swap():
    gt = _scene()
    pred = gt.copy()
    pred[gt == UPPER] = OUTER  # the model calls the whole top a coat
    swap = errors.swapped_regions(pred, gt)
    assert swap[gt == UPPER].all() and not swap[gt != UPPER].any()


def test_zones_add_up_to_the_full_matrix_and_split_errors():
    gt = _scene()
    pred = gt.copy()
    pred[gt == UPPER] = OUTER  # region swap
    pred[13, 9] = BG  # one wrong pixel deep inside pants
    gt[0, 0] = IGNORE
    z = errors.zone_confusions(pred, gt, [0.02, 0.05])
    assert z.shape == (2, 3, NUM_CLASSES, NUM_CLASSES)
    keep = gt != IGNORE
    full = np.bincount(
        gt[keep].astype(int) * NUM_CLASSES + pred[keep], minlength=NUM_CLASSES**2
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    assert (z.sum(1) == full).all()  # every width: zones add up
    edge, swap, inner = z[0]
    assert inner[PANTS, BG] == 1 and swap[UPPER, OUTER] > 0 and edge[UPPER, OUTER] > 0
    assert swap[UPPER, OUTER] + edge[UPPER, OUTER] == (gt == UPPER).sum()
    # a wider band moves swapped pixels into the edge zone, never the other way
    assert z[1, 0, UPPER, OUTER] >= z[0, 0, UPPER, OUTER]
    shares = errors.error_shares(z[0])
    assert sum(shares.values()) == pytest.approx(1.0)
    cls = errors.class_error_shares(z[0])
    assert cls[UPPER].sum() == pytest.approx(1.0) and np.isnan(cls[OUTER]).all()


def test_zone_confusions_rejects_bad_labels():
    gt = _scene()
    with pytest.raises(ValueError):
        errors.zone_confusions(np.full_like(gt, NUM_CLASSES), gt, [0.01])
    with pytest.raises(ValueError):
        errors.zone_confusions(gt[:5], gt, [0.01])


def test_error_shares_without_errors_is_none():
    z = errors.zone_confusions(_scene(), _scene(), [0.01])[0]
    assert errors.error_shares(z) == {"edge": None, "swap": None, "inner": None}
