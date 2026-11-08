"""Multi-person counts (M11): tiny synthetic masks, no data, no GPU."""

import numpy as np
import pandas as pd
import pytest

from cloth_segment import errors, multiperson
from cloth_segment.data.labels import IGNORE, SHARED_CLASSES

BG, UPPER, PANTS = (SHARED_CLASSES.index(c) for c in ("background", "upper", "pants"))


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """40x40: one labelled person (left); the prediction also parses a second one (right)."""
    gt = np.full((40, 40), BG, dtype=np.uint8)
    gt[5:20, 2:12] = UPPER
    gt[20:35, 2:12] = PANTS
    pred = gt.copy()
    pred[5:20, 25:35] = UPPER  # the other person, not touching the labelled one
    pred[20:35, 25:35] = PANTS
    return gt, pred


def test_false_person_splits_edge_from_interior():
    gt, pred = _scene()
    fp, fp_int = multiperson.false_person(pred, gt, width=2)
    assert fp.sum() == 300  # the whole second person
    assert fp_int.sum() == 300  # far from any GT boundary
    pred2 = gt.copy()
    pred2[5:20, 12] = UPPER  # one column grown past the GT edge
    fp, fp_int = multiperson.false_person(pred2, gt, width=2)
    assert fp.sum() == 15 and fp_int.sum() == 0


def test_false_person_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        multiperson.false_person(np.zeros((3, 3)), np.zeros((3, 4)), 1)


def test_separate_person_needs_no_contact_with_gt_person():
    gt, pred = _scene()
    sep = multiperson.separate_person(pred, gt)
    assert sep.sum() == 300 and not sep[:, :20].any()
    pred[20, 12:25] = UPPER  # a bridge to the labelled person
    assert multiperson.separate_person(pred, gt).sum() == 0


def test_blob_ignore_keeps_small_components_and_grows_big_ones():
    gt, pred = _scene()
    pred[1, 38] = UPPER  # a single stray pixel elsewhere
    fp, fp_int = multiperson.false_person(pred, gt, width=2)
    drop = multiperson.blob_ignore(fp, fp_int, width=2, min_px=100)
    assert drop.sum() == 300 and not drop[1, 38]
    assert not multiperson.blob_ignore(fp, fp_int, width=2, min_px=1000).any()
    assert not multiperson.blob_ignore(fp, np.zeros_like(fp), width=2, min_px=1).any()


def test_blob_ignore_takes_the_edge_part_of_a_big_blob():
    gt, pred = _scene()
    gt[5:20, 24] = UPPER  # a GT boundary next to the second person
    fp, fp_int = multiperson.false_person(pred, gt, width=2)
    assert fp_int.sum() < fp.sum()
    drop = multiperson.blob_ignore(fp, fp_int, width=2, min_px=100)
    assert (drop == fp).all()


def test_image_result_matrices_match_evaluate_rule():
    gt, pred = _scene()
    gt[0, 0] = IGNORE
    stats, cms = multiperson.image_result(pred, gt, 0.01, [0.05, 0.5])
    keep = gt != IGNORE
    assert (cms[0] == errors._cm(gt[keep], pred[keep])).all()
    assert cms[0].sum() == keep.sum()
    assert cms[1].sum() == keep.sum() - 300  # 300 px >= 5% of 1600
    assert (cms[2] == cms[0]).all()  # 300 px < 50%
    assert stats["fp_int"] == 300 and stats["fp_int_sep"] == 300
    assert stats["ignored_0.05"] == 300 and stats["fp_int_ignored_0.5"] == 0
    assert stats["person_gt"] == 300
    assert stats["bg"] == 40 * 40 - 300 - 1


def test_image_result_rejects_unknown_labels():
    gt, pred = _scene()
    pred[0, 0] = 40
    with pytest.raises(ValueError):
        multiperson.image_result(pred, gt, 0.01, [0.01])


def test_concentration():
    v = np.array([50, 30, 10, 10] + [0] * 6)
    c = multiperson.concentration(v, [0.1, 0.2, 1.0])
    assert c == {"0.1": 0.5, "0.2": 0.8, "1.0": 1.0}
    assert multiperson.concentration(np.zeros(4), [0.5]) == {"0.5": 0.0}


def _stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "i": range(10),
            "id": [f"im{k}" for k in range(10)],
            "fp_int": [5, 900, 0, 300, 250, 800, 199, 210, 1000, 50],
        }
    )


def test_pick_review_top_then_random_from_the_rest():
    s = multiperson.pick_review(_stats(), top=3, n_random=2, min_px=200, seed=0)
    assert s["group"].tolist() == ["top"] * 3 + ["random"] * 2
    assert s["fp_int"].head(3).tolist() == [1000, 900, 800]
    rnd = s[s["group"] == "random"]
    assert set(rnd["id"]) <= {"im3", "im4", "im7"}
    assert s["n"].tolist() == list(range(5))
    again = multiperson.pick_review(_stats(), top=3, n_random=2, min_px=200, seed=0)
    assert again.equals(s)
    assert len(multiperson.pick_review(_stats(), 3, 10, 200, 0)) == 6  # only 3 left


def test_tally_verdicts_counts_and_checks_the_sample():
    s = multiperson.pick_review(_stats(), top=2, n_random=1, min_px=200, seed=0)
    v = s[["n", "id"]].assign(verdict=["other_person", "unclear", "other_person"])
    t = multiperson.tally_verdicts(v, s, ["other_person", "unclear", "object_clutter"])
    row = t.set_index("verdict")
    assert row.loc["other_person", "top"] == 1 and row.loc["other_person", "random"] == 1
    assert row.loc["object_clutter", "total"] == 0
    assert row.loc["unclear", "fp_int"] == 900
    with pytest.raises(ValueError):
        multiperson.tally_verdicts(v.iloc[:2], s, ["other_person", "unclear"])
    with pytest.raises(ValueError):
        multiperson.tally_verdicts(v.assign(verdict="nope"), s, ["other_person"])
