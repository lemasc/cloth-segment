import numpy as np
import pytest

from cloth_segment.data import splits as S
from cloth_segment.data.dedup import dhash, near_pairs, thumb_corr, to_u64


def test_group_keys():
    assert S.group_key("lip", "484151_962996") == S.group_key("lip", "484151_1750729")
    assert S.group_key("cihp", "0000006") != S.group_key("cihp", "0000007")
    assert S.group_key("dfmm", "MEN-Denim-id_00000080-01_7_additional") == S.group_key(
        "dfmm", "MEN-Suiting-id_00000080-02_1_front"
    )


def test_duplicate_thresholds():
    assert S.is_duplicate("lip", "cihp", 0.75)
    assert not S.is_duplicate("dfmm", "dfmm", 0.97)
    assert not S.is_duplicate("dfmm", "cihp", 0.75)
    assert S.is_duplicate("dfmm", "dfmm", 0.99)


def test_union_find_is_order_independent():
    a, b = S.UnionFind(), S.UnionFind()
    for x, y in [("c", "b"), ("b", "a")]:
        a.union(x, y)
    for x, y in [("a", "b"), ("b", "c")]:
        b.union(x, y)
    assert a.find("c") == b.find("c") == "a"


def test_unit_hash_deterministic_and_uniform():
    assert S.unit_hash("x") == S.unit_hash("x") != S.unit_hash("x", seed=1)
    u = np.array([S.unit_hash(str(i)) for i in range(5000)])
    assert 0.45 < u.mean() < 0.55 and u.min() >= 0 and u.max() < 1


def test_assign_rules():
    ds = ["lip", "lip", "lip", "cihp", "dfmm"]
    src = ["val", "train", "train", "train", "all"]
    groups = ["g1", "g1", "g2", "g3", "g4"]
    scope = ["", "", "empty_mask", "", ""]
    split, reason = S.assign(ds, src, groups, scope)
    assert split[0] == "test"  # official val is test
    assert (split[1], reason[1]) == ("excluded", "leak_test")  # shares a group with test
    assert (split[2], reason[2]) == ("excluded", "empty_mask")
    assert split[3] in ("train", "val") and split[4] in ("train", "val", "test")


def test_assign_rejects_mixed_groups():
    with pytest.raises(ValueError, match="mixes"):
        S.assign(["lip", "dfmm"], ["train", "all"], ["g", "g"], ["", ""])


def test_dhash_finds_rescaled_copies():
    from PIL import Image

    rng = np.random.default_rng(0)
    base = (rng.random((8, 8)) * 255).astype(np.uint8).repeat(16, 0).repeat(16, 1)
    img = Image.fromarray(base)
    other = Image.fromarray((rng.random((128, 128)) * 255).astype(np.uint8))
    h = to_u64(np.stack([dhash(img), dhash(img.resize((64, 64))), dhash(other)]))
    pairs = near_pairs(h, max_dist=4)
    assert [tuple(p[:2]) for p in pairs] == [(0, 1)]
    t = np.asarray(img.resize((32, 32)))[None]
    assert thumb_corr(t, t)[0] > 0.99


def test_scope_reasons():
    h = np.array([100, 63, 64, 500, 100, 20])
    w = np.array([100, 200, 64, 83, 600, 20])
    empty = np.array([False, False, False, False, False, True])
    got = S.scope_reasons(h, w, empty, min_side=64, max_aspect=6.0)
    # 63 px short side; exactly 64 kept; 500/83 > 6; 600/100 == 6 kept; empty wins
    assert got == ["", "too_small", "", "extreme_aspect", "", "empty_mask"]


def test_out_of_scope_test_image_does_not_leak_into_train():
    # g1: official val image is too small (excluded); its group-mate in train must stay.
    ds, src, groups = ["lip", "lip"], ["val", "train"], ["g1", "g1"]
    split, reason = S.assign(ds, src, groups, ["too_small", ""])
    assert (split[0], reason[0]) == ("excluded", "too_small")
    assert split[1] in ("train", "val")


def test_out_of_scope_rows_do_not_join_duplicate_groups():
    import pandas as pd

    # row 1 (thin) is a duplicate of rows 0 and 2; if it bridged them, train row 2
    # would be dropped as a leak of test row 0.
    df = pd.DataFrame(
        {
            "dataset": ["lip", "lip", "cihp"],
            "source_split": ["val", "train", "train"],
            "id": ["1_1", "2_1", "9"],
            "h": [300, 700, 300],
            "w": [200, 100, 200],
            "c1": [5, 5, 5],
        }
    )
    groups, split, reason = S.apply_rules(df, [0, 1], [1, 2])
    assert (split[1], reason[1]) == ("excluded", "extreme_aspect")
    assert groups[0] != groups[2] and split[2] in ("train", "val")
