import math

import pytest
import torch

from cloth_segment.evaluate import EVAL_VERSION, confusion_matrix, restore_logits, scores
from cloth_segment.lib.data.labels import IGNORE, NUM_CLASSES

NAMES3 = ("bg", "a", "b")


def test_perfect_prediction():
    t = torch.tensor([[0, 1], [2, 1]])
    s = scores(confusion_matrix(t, t, 3), NAMES3)
    assert s["miou"] == 1.0 and s["mdice"] == 1.0 and s["pixel_acc"] == 1.0
    assert s["eval_version"] == EVAL_VERSION


def test_hand_computed_example():
    # gt:   0 0 1 1 1     pred: 0 1 1 1 0
    t = torch.tensor([0, 0, 1, 1, 1])
    p = torch.tensor([0, 1, 1, 1, 0])
    cm = confusion_matrix(p, t, 2)
    assert cm.tolist() == [[1, 1], [1, 2]]
    s = scores(cm, ("bg", "a"))
    # class 0: tp1 fp1 fn1 -> iou 1/3, dice 2/4; class 1: tp2 fp1 fn1 -> iou 2/4, dice 4/6
    assert s["iou"] == {"bg": pytest.approx(1 / 3), "a": pytest.approx(0.5)}
    assert s["dice"] == {"bg": pytest.approx(0.5), "a": pytest.approx(2 / 3)}
    assert s["miou"] == pytest.approx((1 / 3 + 0.5) / 2)
    assert s["pixel_acc"] == pytest.approx(3 / 5)


def test_dice_iou_relation():
    g = torch.Generator().manual_seed(0)
    t = torch.randint(0, 3, (500,), generator=g)
    p = torch.randint(0, 3, (500,), generator=g)
    s = scores(confusion_matrix(p, t, 3), NAMES3)
    for n in NAMES3:
        iou = s["iou"][n]
        assert s["dice"][n] == pytest.approx(2 * iou / (1 + iou))


def test_ignore_pixels_are_skipped():
    t = torch.tensor([0, 1, IGNORE, IGNORE])
    p = torch.tensor([0, 1, 2, 0])  # predictions on ignore pixels do not count
    cm = confusion_matrix(p, t, 3)
    assert cm.sum() == 2
    assert scores(cm, NAMES3)["miou"] == 1.0


def test_absent_class_not_in_mean():
    # class "b" has no ground truth and no predictions -> None, not in mean
    t = torch.tensor([0, 0, 1, 1])
    s = scores(confusion_matrix(t, t, 3), NAMES3)
    assert s["iou"]["b"] is None and s["dice"]["b"] is None
    assert s["in_mean"] == ["bg", "a"]
    assert s["miou"] == 1.0


def test_false_positive_only_class():
    # "b" is predicted but absent from ground truth: iou 0, still not in the mean,
    # but its false positives lower the true class's IoU.
    t = torch.tensor([0, 0, 1, 1])
    p = torch.tensor([0, 0, 1, 2])
    s = scores(confusion_matrix(p, t, 3), NAMES3)
    assert s["iou"]["b"] == 0.0
    assert s["in_mean"] == ["bg", "a"]
    assert s["iou"]["a"] == 0.5
    assert s["miou"] == pytest.approx(0.75)


def test_accumulation_is_dataset_level():
    # Summing matrices != averaging per-image mIoU.
    t1, p1 = torch.tensor([0] * 99 + [1]), torch.tensor([0] * 100)
    t2, p2 = torch.tensor([1, 1]), torch.tensor([1, 1])
    cm = confusion_matrix(p1, t1, 2) + confusion_matrix(p2, t2, 2)
    s = scores(cm, ("bg", "a"))
    assert s["iou"]["a"] == pytest.approx(2 / 3)
    assert s["gt_pixels"] == {"bg": 99, "a": 3}


@pytest.mark.parametrize("bad", [12, -1, 254])
def test_invalid_target_raises(bad):
    t = torch.tensor([0, bad])
    with pytest.raises(ValueError, match="target"):
        confusion_matrix(torch.tensor([0, 0]), t)


def test_invalid_pred_and_shape_raise():
    with pytest.raises(ValueError, match="pred has"):
        confusion_matrix(torch.tensor([NUM_CLASSES]), torch.tensor([0]))
    with pytest.raises(ValueError, match="!="):
        confusion_matrix(torch.zeros(2, 2), torch.zeros(4))


def test_scores_shape_mismatch_raises():
    with pytest.raises(ValueError):
        scores(torch.zeros(3, 3), ("a", "b"))


def test_empty_matrix():
    s = scores(torch.zeros(3, 3, dtype=torch.long), NAMES3)
    assert s["miou"] is None and s["pixel_acc"] is None and s["in_mean"] == []


def test_restore_logits_ignores_padding():
    # 2 classes, 8x8 canvas, valid 4x8 (landscape image padded at the bottom).
    logits = torch.zeros(2, 8, 8)
    logits[1, :4, :] = 1.0  # valid region says class 1
    logits[0, 4:, :] = 9.0  # padding says class 0 strongly; must not leak in
    out = restore_logits(logits, valid_hw=(4, 8), orig_hw=(20, 40))
    assert out.shape == (2, 20, 40)
    assert (out.argmax(0) == 1).all()


def test_restore_logits_validates():
    with pytest.raises(ValueError):
        restore_logits(torch.zeros(1, 2, 8, 8), (8, 8), (8, 8))
    with pytest.raises(ValueError):
        restore_logits(torch.zeros(2, 8, 8), (9, 8), (8, 8))


def test_restore_logits_identity_when_same_size():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 6, 6, generator=g)
    out = restore_logits(logits, (6, 6), (6, 6))
    assert torch.allclose(out, logits)
    assert not math.isnan(float(out.sum()))
