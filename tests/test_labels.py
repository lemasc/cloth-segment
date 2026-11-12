import numpy as np
import pytest

from cloth_segment.lib.data.labels import (
    IGNORE,
    MAPPING,
    NUM_CLASSES,
    SHARED_CLASSES,
    UnknownLabelError,
    colorize,
    convert,
    label_hash,
)
from cloth_segment.lib.data.sources import RAW_LABELS


def name_of(ds, raw_name):
    rid = {n: i for i, n in RAW_LABELS[ds].items()}[raw_name]
    sid = MAPPING[ds][rid]
    return "ignore" if sid == IGNORE else SHARED_CLASSES[sid]


@pytest.mark.parametrize("ds", list(RAW_LABELS))
def test_every_raw_id_is_mapped_to_a_valid_target(ds):
    assert set(MAPPING[ds]) == set(RAW_LABELS[ds])
    assert all(v == IGNORE or 0 <= v < NUM_CLASSES for v in MAPPING[ds].values())
    assert MAPPING[ds][0] == 0


def test_every_shared_class_is_used_by_every_dataset():
    # A class missing from a dataset would make it unlearnable from that source.
    for ds, m in MAPPING.items():
        assert set(range(NUM_CLASSES)) <= set(m.values()), ds


@pytest.mark.parametrize(
    "ds,raw,expected",
    [
        ("lip", "jumpsuits", "dress"),
        ("cihp", "torso_skin", "skin"),
        ("lip", "socks", "footwear"),
        ("cihp", "left_arm", "skin"),
        ("dfmm", "leggings", "footwear"),
        ("dfmm", "socks", "footwear"),
        ("dfmm", "rompers", "dress"),
        ("dfmm", "belt", "ignore"),
        ("dfmm", "bag", "ignore"),
        ("dfmm", "ring", "ignore"),
    ],
)
def test_decisions(ds, raw, expected):
    assert name_of(ds, raw) == expected


@pytest.mark.parametrize("ds", ["lip", "cihp"])
def test_left_right_merged(ds):
    for part in ("arm", "leg", "shoe"):
        assert name_of(ds, f"left_{part}") == name_of(ds, f"right_{part}")


def test_convert_maps_values_and_keeps_shape():
    raw = np.array(
        [[0, 14, 15], [10, 6, 12]], dtype=np.uint8
    )  # bg, face, skin; belt, leggings, bag
    out = convert(raw, "dfmm")
    assert out.dtype == np.uint8 and out.shape == raw.shape
    s = SHARED_CLASSES.index
    assert out.tolist() == [[0, s("face"), s("skin")], [IGNORE, s("footwear"), IGNORE]]


def test_convert_is_flip_equivariant():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 20, size=(32, 24), dtype=np.uint8)
    assert np.array_equal(convert(raw[:, ::-1], "lip"), convert(raw, "lip")[:, ::-1])


@pytest.mark.parametrize("ds,bad", [("lip", 20), ("cihp", 255), ("dfmm", 24)])
def test_convert_rejects_unknown_values(ds, bad):
    raw = np.zeros((4, 4), dtype=np.uint8)
    raw[1, 1] = bad
    with pytest.raises(UnknownLabelError, match=str(bad)):
        convert(raw, ds)


def test_convert_rejects_rgb():
    with pytest.raises(ValueError, match="2-D"):
        convert(np.zeros((4, 4, 3), dtype=np.uint8), "lip")


def test_colorize_and_hash():
    m = np.array([[0, IGNORE]], dtype=np.uint8)
    assert colorize(m).shape == (1, 2, 3)
    assert len(label_hash()) == 12 and label_hash() == label_hash()
