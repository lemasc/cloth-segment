import numpy as np
import pandas as pd
import pytest

from cloth_segment import paths
from cloth_segment.data.labels import IGNORE, SHARED_CLASSES
from cloth_segment.review import (
    IGNORE_COLOUR,
    PALETTE,
    RAW_PALETTE,
    colour_mask,
    crop_box,
    load_review_config,
    outline,
    overlay,
    per_image_recall,
    pick_samples,
    raw_legend,
    raw_legend_html,
    read_verdicts,
    size_table,
    split_count,
    tally,
    write_verdicts,
)


def _raw_shared():
    raw = pd.DataFrame(
        {
            "index": [0, 0, 1, 2],
            "id": ["a", "a", "b", "c"],
            "raw_name": ["left_shoe", "right_shoe", "left_shoe", "socks"],
            "raw_px": [10, 30, 50, 8],
            "hit_px": [5, 15, 50, 0],
        }
    )
    shared = pd.DataFrame(
        {"index": [0, 0, 1, 2], "h": [10, 10, 20, 4], "w": [10, 10, 10, 4], "cls": list("wxyz")}
    )
    return raw, shared


def test_per_image_recall_sums_raw_classes_per_image():
    raw, shared = _raw_shared()
    df = per_image_recall(raw, shared, ["left_shoe", "right_shoe"]).set_index("index")
    assert list(df.index) == [0, 1]
    assert df.loc[0, "raw_px"] == 40 and df.loc[0, "hit_px"] == 20
    assert df.loc[0, "recall"] == 0.5 and df.loc[0, "share"] == 40 / 100
    assert df.loc[1, "recall"] == 1.0 and df.loc[1, "share"] == 50 / 200


def test_per_image_recall_rejects_ignored_raw_class():
    raw, shared = _raw_shared()
    raw.loc[3, "hit_px"] = np.nan
    with pytest.raises(ValueError, match="IGNORE"):
        per_image_recall(raw, shared, ["socks"])


def test_pick_samples_worst_then_random_without_overlap():
    df = pd.DataFrame(
        {
            "index": range(10),
            "recall": [0.9, 0.1, 0.5, 0.1, 0.3, 0.8, 0.2, 0.7, 0.6, 0.4],
            "raw_px": [5, 5, 5, 9, 5, 5, 5, 5, 5, 5],
            "share": [0.01] * 9 + [0.0001],  # index 9 is too small
        }
    )
    p = pick_samples(df, n_worst=3, n_random=4, min_share=0.001, seed=0)
    worst = p[p["pick"] == "worst"]["index"].tolist()
    assert worst == [3, 1, 6]  # tie at 0.1 broken by larger area
    rnd = p[p["pick"] == "random"]["index"].tolist()
    assert len(rnd) == 4 and not set(rnd) & set(worst) and 9 not in rnd
    again = pick_samples(df, n_worst=3, n_random=4, min_share=0.001, seed=0)
    assert again.equals(p)  # same seed, same picks


def test_pick_samples_random_capped_by_candidates():
    df = pd.DataFrame({"index": [0, 1], "recall": [0.1, 0.2], "raw_px": [1, 1], "share": [1, 1]})
    p = pick_samples(df, n_worst=1, n_random=5, min_share=0, seed=0)
    assert p["pick"].tolist() == ["worst", "random"]


def test_split_count():
    assert split_count(12, 2) == [6, 6]
    assert split_count(7, 3) == [3, 2, 2]


def test_size_table_groups_and_recall():
    df = pd.DataFrame(
        {
            "index": [0, 1, 2],
            "share": [0.001, 0.003, 0.02],
            "raw_px": [10, 10, 20],
            "hit_px": [0, 5, 20],
        }
    )
    t = size_table(df, [0, 0.002, 0.01, 1], ["small", "mid", "big"])
    assert t["images"].tolist() == [1, 1, 1]
    assert t["recall"].tolist() == [0.0, 0.5, 1.0]
    empty = size_table(df, [0, 0.0005, 1], ["none", "all"])
    assert np.isnan(empty.loc["none", "recall"])


def test_colour_and_overlay():
    mask = np.array([[0, 4], [IGNORE, 9]], dtype=np.uint8)
    col = colour_mask(mask)
    assert (col[0, 1] == PALETTE[4]).all() and (col[1, 0] == IGNORE_COLOUR).all()
    img = np.full((2, 2, 3), 200, dtype=np.uint8)
    out = overlay(img, mask, alpha=0.5)
    assert (out[0, 0] == 200).all()  # background keeps the photo
    assert (out[0, 1] == np.round((200 + PALETTE[4].astype(float)) / 2)).all()


def test_palette_has_one_colour_per_class():
    assert len(PALETTE) == len(SHARED_CLASSES)
    assert len({tuple(c) for c in PALETTE}) == len(PALETTE)


def test_outline_is_inner_edge():
    region = np.zeros((7, 7), dtype=bool)
    region[1:6, 1:6] = True
    e = outline(region, width=1)
    assert e[1, 1] and e[1, 3] and not e[3, 3] and not e[0, 0]
    assert e.sum() == 16


def test_crop_box_grows_and_clips():
    region = np.zeros((300, 200), dtype=bool)
    region[100:110, 50:60] = True
    y0, y1, x0, x1 = crop_box(region, margin=0.5, min_side=96)
    assert y1 - y0 == 96 and x1 - x0 == 96
    region[:] = False
    region[0:5, 0:5] = True
    assert crop_box(region, min_side=96)[:3:2] == (0, 0)
    assert crop_box(np.zeros((4, 5), dtype=bool)) == (0, 4, 0, 5)


def test_verdicts_round_trip_and_unchanged(tmp_path):
    p = tmp_path / "v.csv"
    assert read_verdicts(p).empty
    df = pd.DataFrame(
        {
            "question": ["q3", "q1", "q1"],
            "key": ["lip:5", "lip:2", "lip:9"],
            "verdict": ["Y", "", ""],
            "note": ["", "looks odd", ""],
        }
    )
    assert write_verdicts(p, df) is True
    back = read_verdicts(p)
    assert back["key"].tolist() == ["lip:2", "lip:5"]  # empty row dropped, sorted
    assert write_verdicts(p, df) is False  # same content, file untouched


def test_tally_counts_by_pick_and_dataset():
    samples = pd.DataFrame(
        {
            "key": ["lip:1", "cihp:2", "lip:3"],
            "pick": ["worst", "random", "random"],
            "dataset": ["lip", "cihp", "lip"],
        }
    )
    v = pd.DataFrame(
        {
            "question": ["q"] * 3,
            "key": ["lip:1", "cihp:2", "lip:3"],
            "verdict": ["A", "A", ""],
            "note": [""] * 3,
        }
    )
    t = tally(v, samples, {"A": "a", "B": "b"}).set_index("verdict")
    assert t.loc["A", "all"] == 2 and t.loc["A", "worst"] == 1 and t.loc["A", "random"] == 1
    assert t.loc["A", "lip"] == 1 and t.loc["A", "cihp"] == 1 and t.loc["B", "all"] == 0


def test_m10_review_config_is_valid():
    cfg = load_review_config(paths.CONFIGS / "review" / "m10.yaml")
    assert cfg["rank_run"] in cfg["runs"]
    for q in cfg["questions"].values():
        assert set(q["datasets"]) <= {"lip", "cihp", "dfmm"} and q["verdicts"]


def test_review_config_missing_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("out_dir: x\nruns: {A: r}\nrank_run: A\nseed: 0\nquestions: {q: {title: t}}\n")
    with pytest.raises(ValueError, match="misses"):
        load_review_config(p)


def test_raw_palette_covers_every_raw_class():
    from cloth_segment.data.sources import RAW_LABELS

    assert max(max(v) for v in RAW_LABELS.values()) < len(RAW_PALETTE)
    assert len({tuple(c) for c in RAW_PALETTE}) == len(RAW_PALETTE)


def test_overlay_raw_uses_raw_palette():
    img = np.zeros((1, 2, 3), dtype=np.uint8)
    out = overlay(img, np.array([[0, 10]], dtype=np.uint8), alpha=1.0, raw=True)
    assert (out[0, 0] == 0).all() and (out[0, 1] == RAW_PALETTE[10]).all()


def test_raw_legend_names_target_and_share():
    # LIP: 10 jumpsuits -> dress, 0 background
    raw = np.array([[10, 10, 10, 0]], dtype=np.uint8)
    items = raw_legend(raw, "lip")
    assert items[0] == {"raw_id": 10, "raw": "jumpsuits", "shared": "dress", "share": 0.75}
    html = raw_legend_html(items)
    assert "jumpsuits → <b>dress</b> (75.0%)" in html and "background →" not in html


def test_raw_legend_ignored_class():
    # DFMM: 10 belt -> ignored
    items = raw_legend(np.array([[10]], dtype=np.uint8), "dfmm")
    assert items[0]["shared"] is None and "ignored" in raw_legend_html(items)
