import torch

from cloth_segment.data.labels import IGNORE, SHARED_CLASSES
from cloth_segment.rawclass import (
    ACCESSORY,
    RawClassTally,
    accessory_raw_ids,
    image_counts,
)

FACE = SHARED_CLASSES.index("face")
BG = 0


def test_accessory_raw_ids_per_dataset():
    assert set(accessory_raw_ids("lip").values()) == {"glove", "sunglasses", "scarf"}
    assert "eyeglass" in accessory_raw_ids("dfmm").values()
    assert "belt" not in accessory_raw_ids("dfmm").values()  # ignored, not accessory


def test_counts_recall_and_false_positives():
    # LIP: glove=3, scarf=11. Row 0: glove pixels predicted accessory, accessory, face, bg.
    raw = torch.tensor([[3, 3, 3, 3], [11, 11, 0, 0]])
    target = torch.tensor([[ACCESSORY] * 4, [ACCESSORY, ACCESSORY, BG, BG]])
    pred = torch.tensor([[ACCESSORY, ACCESSORY, FACE, BG], [ACCESSORY, ACCESSORY, ACCESSORY, BG]])
    per_raw, fp = image_counts(pred, target, raw, [3, 11])
    assert fp == 1  # the accessory prediction on a background pixel
    t = RawClassTally({3: "glove", 11: "scarf"})
    t.add(per_raw, fp)
    r = t.result()["raw_classes"]
    assert r["glove"]["recall"] == 0.5
    assert r["glove"]["where"] == {"background": 1, "face": 1, "accessory": 2}
    assert r["glove"]["iou_others_ignored"] == 2 / (4 + 1)
    assert r["scarf"]["recall"] == 1.0
    assert r["glove"]["image_hit_rate"] == 1.0  # exactly half counts as a hit


def test_ignore_pixels_are_not_false_positives():
    raw = torch.tensor([[0, 0]])
    target = torch.tensor([[IGNORE, BG]])
    pred = torch.tensor([[ACCESSORY, ACCESSORY]])
    _, fp = image_counts(pred, target, raw, [3])
    assert fp == 1


def test_class_absent_has_no_rates_and_no_division():
    t = RawClassTally({3: "glove"})
    t.add({3: torch.zeros(len(SHARED_CLASSES), dtype=torch.int64)}, 0)
    r = t.result()["raw_classes"]["glove"]
    assert r["gt_pixels"] == 0 and r["recall"] is None and r["image_hit_rate"] is None


def test_headwear_summary_is_pixel_weighted():
    import pandas as pd

    from cloth_segment.rawclass import headwear_summary

    cols = {f"pred_{c}": 0.0 for c in SHARED_CLASSES}
    a = pd.DataFrame(
        [
            {"sunglasses_px": 100, "hat_px_gt": 50} | cols | {"pred_headwear": 1.0},
            {"sunglasses_px": 300, "hat_px_gt": 0} | cols | {"pred_face": 1.0},
        ]
    )
    s = headwear_summary(a)
    assert s["pixel_share_all"]["headwear"] == 0.25
    assert s["by_gt_hat"]["with_gt_hat"]["headwear"] == 1.0
    assert s["by_gt_hat"]["without_gt_hat"]["headwear"] == 0.0
    assert s["by_gt_hat"]["without_gt_hat"]["face"] == 1.0
    assert s["by_gt_hat"]["with_gt_hat"]["images"] == 1
    assert s["images_over_half_headwear"] == 1 and s["images_over_half_headwear_with_gt_hat"] == 1
