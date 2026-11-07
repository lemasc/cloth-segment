import torch

from cloth_segment.data.labels import IGNORE, SHARED_CLASSES
from cloth_segment.data.sources import RAW_LABELS
from cloth_segment.rawclass import (
    ACCESSORY,
    RawClassTally,
    accessory_raw_ids,
    image_counts,
)

FACE = SHARED_CLASSES.index("face")
DRESS = SHARED_CLASSES.index("dress")
UPPER = SHARED_CLASSES.index("upper")
PANTS = SHARED_CLASSES.index("pants")
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


def test_box_stats_fill_and_around_share():
    import numpy as np

    from cloth_segment.rawclass import box_stats

    # 4x4 mask: raw 5 (glasses) at two corners of a 2x3 box; raw 2 (face) inside the box.
    raw = np.zeros((4, 4), dtype=np.uint8)
    raw[1, 1] = raw[2, 3] = 5
    raw[1, 2] = raw[2, 1] = 2
    s = box_stats(raw, rid=5, around=2)
    assert s["pixels"] == 2 and (s["box_h"], s["box_w"]) == (2, 3)
    assert s["fill"] == 2 / 6 and s["around_share"] == 2 / 6
    assert box_stats(raw, rid=9, around=2) is None


def test_accessory_supply_skips_excluded_and_ignore():
    import pandas as pd

    from cloth_segment.rawclass import accessory_supply

    def row(split, acc, other, ignore=0):
        r = {"dataset": "dfmm", "split": split} | {f"c{i}": 0 for i in range(len(SHARED_CLASSES))}
        return r | {f"c{ACCESSORY}": acc, "c0": other, "c255": ignore}

    m = pd.DataFrame(
        [row("train", 10, 90, ignore=500), row("train", 0, 100), row("excluded", 50, 50)]
    )
    t = accessory_supply(m).set_index("split")
    assert list(t.index) == ["train"]
    r = t.loc["train"]
    assert r["images"] == 2 and r["images_with_accessory"] == 1 and r["pct_images"] == 50.0
    assert r["accessory_pixels"] == 10 and r["pct_labelled_pixels"] == 100 * 10 / 200


def test_miou_without_uses_present_classes_only():
    from cloth_segment.rawclass import miou_without

    names = ["a", "b", "absent", "accessory"]
    cm = [[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 0, 0], [0, 1, 0, 1]]
    m = {"miou": 0.6, "iou": {"a": 1.0, "b": 0.5, "absent": 0.0, "accessory": 0.3}}
    m["confusion_matrix"] = cm
    assert list(m["iou"]) == names
    r = miou_without(m)
    assert r["miou_without_accessory"] == 0.75  # mean of a and b; `absent` has no GT pixels
    assert r["accessory"] == 0.3 and r["miou_all"] == 0.6


def test_raw_pred_matrix_counts_every_pixel():
    from cloth_segment.rawclass import RAW_MAX, raw_pred_matrix

    raw = torch.tensor([[6, 6, 10], [10, 10, 0]], dtype=torch.uint8)  # LIP dress, jumpsuits
    pred = torch.tensor([[DRESS, UPPER, DRESS], [UPPER, PANTS, BG]])
    m = raw_pred_matrix(pred, raw)
    assert m.shape == (RAW_MAX, len(SHARED_CLASSES))
    assert int(m.sum()) == 6
    assert m[6, DRESS] == 1 and m[6, UPPER] == 1
    assert m[10, DRESS] == 1 and m[10, UPPER] == 1 and m[10, PANTS] == 1
    assert m[0, BG] == 1


def test_shared_counts_skip_ignore_and_match_confusion_matrix():
    from cloth_segment.evaluate import confusion_matrix
    from cloth_segment.rawclass import shared_counts

    target = torch.tensor([[DRESS, DRESS, IGNORE], [UPPER, BG, BG]])
    pred = torch.tensor([[DRESS, UPPER, DRESS], [UPPER, BG, DRESS]])
    sc = shared_counts(pred, target)
    cm = confusion_matrix(pred, target, len(SHARED_CLASSES))
    assert torch.equal(sc, torch.stack([cm.sum(1), cm.sum(0), cm.diagonal()]))
    assert sc[0, DRESS] == 2 and sc[1, DRESS] == 2 and sc[2, DRESS] == 1


def test_raw_rows_hit_uses_mapped_class_and_ignore_has_none():
    from cloth_segment.data.labels import MAPPING
    from cloth_segment.rawclass import raw_pred_matrix, raw_rows

    belt = next(r for r, n in RAW_LABELS["dfmm"].items() if n == "belt")
    assert MAPPING["dfmm"].get(belt, IGNORE) == IGNORE
    dress = next(r for r, n in RAW_LABELS["dfmm"].items() if n == "dress")
    raw = torch.tensor([[dress, dress, belt]], dtype=torch.uint8)
    pred = torch.tensor([[DRESS, UPPER, UPPER]])
    rows = {r["raw_name"]: r for r in raw_rows(raw_pred_matrix(pred, raw), "dfmm")}
    assert rows["belt"] == {
        "raw_id": belt,
        "raw_name": "belt",
        "shared": None,
        "raw_px": 1,
        "hit_px": None,
    }
    assert rows["dress"]["shared"] == "dress"
    assert rows["dress"]["raw_px"] == 2 and rows["dress"]["hit_px"] == 1


def test_raw_rows_rejects_unknown_raw_id():
    import pytest

    from cloth_segment.rawclass import raw_pred_matrix, raw_rows

    raw = torch.tensor([[200]], dtype=torch.uint8)
    with pytest.raises(ValueError, match="not a known raw class"):
        raw_rows(raw_pred_matrix(torch.tensor([[BG]]), raw), "lip")


def test_existing_breakdown_lists_only_present_files(tmp_path):
    from cloth_segment.rawclass import existing_breakdown

    assert existing_breakdown(tmp_path, ["run-a"], ["lip", "cihp"]) == []
    d = tmp_path / "run-a" / "cihp"
    d.mkdir(parents=True)
    (d / "per_image_raw.csv").write_text("x")
    (d / "other.txt").write_text("x")
    assert existing_breakdown(tmp_path, ["run-a"], ["lip", "cihp"]) == [d / "per_image_raw.csv"]
    assert existing_breakdown(tmp_path, ["run-b"], ["cihp"]) == []
