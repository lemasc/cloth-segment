import pandas as pd
import pytest

from cloth_segment.data.composition import merge_composition
from cloth_segment.data.labels import IGNORE, MAPPING, SHARED_CLASSES
from cloth_segment.data.sources import RAW_LABELS


def fake_counts(ds, **by_name):
    """Two images; raw class `name` gets the given pixels in image 1 and half in image 2."""
    ids = {n: i for i, n in RAW_LABELS[ds].items()}
    data = {f"v{i}": [0, 0] for i in RAW_LABELS[ds]}
    for name, px in by_name.items():
        data[f"v{ids[name]}"] = [px, px // 2]
    return pd.DataFrame(data)


def test_accessory_split_in_lip():
    t = merge_composition(fake_counts("lip", glove=100, sunglasses=100, scarf=200), "lip")
    acc = t[t.shared == "accessory"].set_index("raw")
    assert acc.loc["scarf", "pct_of_shared"] == pytest.approx(100 * 300 / 600)
    assert acc.loc["glove", "pct_of_shared"] == pytest.approx(100 * 150 / 600)
    assert acc["pct_of_shared"].sum() == pytest.approx(100)
    assert acc.loc["glove", "images"] == 2


def test_percentages_sum_and_ignore_row():
    t = merge_composition(fake_counts("dfmm", top=50, belt=30, eyeglass=20), "dfmm")
    assert t["pct_of_dataset"].sum() == pytest.approx(100)
    belt = t[t.raw == "belt"].iloc[0]
    assert belt.shared == "ignore"
    assert belt.pct_of_shared == pytest.approx(100)
    assert t.groupby("shared")["pct_of_shared"].sum().round(6).isin([0, 100]).all()


@pytest.mark.parametrize("ds", list(RAW_LABELS))
def test_one_row_per_raw_class_matching_the_mapping(ds):
    t = merge_composition(fake_counts(ds), ds)
    assert len(t) == len(RAW_LABELS[ds])
    for r in t.itertuples():
        target = MAPPING[ds][r.raw_id]
        assert r.shared == ("ignore" if target == IGNORE else SHARED_CLASSES[target])
    assert (t.pixels == 0).all()
    assert (t.pct_of_shared == 0).all()  # empty input must not produce NaN


def test_missing_column_fails_loudly():
    with pytest.raises(KeyError):
        merge_composition(fake_counts("lip").drop(columns="v3"), "lip")
