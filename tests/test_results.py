"""T9 results from files: rebuilt and adjusted matrices, per-image counts, bootstrap,
tables and figures. Tiny synthetic inputs; no data, no GPU."""

import numpy as np
import pandas as pd
import pytest
import torch

from cloth_segment import results
from cloth_segment.data.labels import NUM_CLASSES, SHARED_CLASSES
from cloth_segment.evaluate import scores

BG, DRESS, UPPER = (SHARED_CLASSES.index(c) for c in ("background", "dress", "upper"))


def _raw_conf(scale: int = 1) -> dict:
    """background, a real dress class and a jumpsuit class that also maps to dress."""
    return {
        "checkpoint_step": 8,
        "raw_classes": {
            "background": {
                "shared": "background",
                "where": {"background": 90 * scale, "dress": 10},
            },
            "dress": {"shared": "dress", "where": {"dress": 30, "upper": 10, "background": 0}},
            "jumpsuits": {"shared": "dress", "where": {"dress": 5, "upper": 15}, "recall": 0.25},
            "belt": {"shared": None, "where": {"background": 7}},
        },
    }


def test_cm_from_raw_sums_rows_and_skips_ignored_raw_classes():
    cm = results.cm_from_raw(_raw_conf())
    assert cm[BG, BG] == 90 and cm[BG, DRESS] == 10
    assert cm[DRESS, DRESS] == 35 and cm[DRESS, UPPER] == 25
    assert cm.sum() == 160  # belt (maps to IGNORE) is not counted


def test_cm_from_raw_drop_removes_only_that_raw_class():
    full, adj = results.cm_from_raw(_raw_conf()), results.cm_from_raw(_raw_conf(), ["jumpsuits"])
    diff = full - adj
    assert diff[DRESS, DRESS] == 5 and diff[DRESS, UPPER] == 15 and diff.sum() == 20


def test_cm_from_raw_rejects_unknown_or_ignored_drop():
    with pytest.raises(ValueError, match="not in this raw_confusion"):
        results.cm_from_raw(_raw_conf(), ["socks"])
    with pytest.raises(ValueError, match="maps to IGNORE"):
        results.cm_from_raw(_raw_conf(), ["belt"])


def _counts_from_cm(cm: np.ndarray) -> np.ndarray:
    return np.stack([cm.sum(1), cm.sum(0), np.diag(cm)])


def test_miou_from_counts_matches_evaluate_scores():
    rng = np.random.default_rng(1)
    cm = rng.integers(0, 50, size=(NUM_CLASSES, NUM_CLASSES))
    cm[3] = 0  # a class without ground truth is not in the mean
    ref = scores(torch.from_numpy(cm))
    assert results.miou_from_counts(_counts_from_cm(cm)) == pytest.approx(ref["miou"])
    keep = results.keep_mask(["accessory"])
    assert results.miou_from_counts(_counts_from_cm(cm), keep) == pytest.approx(
        results.mean_without(ref, "iou", ["accessory"])
    )


def test_keep_mask_rejects_unknown_class():
    with pytest.raises(ValueError):
        results.keep_mask(["shorts"])


def _write_csv(path, rows):
    pd.DataFrame(rows, columns=["index", "id", "h", "w", "cls", "gt", "pred", "tp"]).to_csv(
        path, index=False
    )


def test_read_image_counts_fills_missing_classes(tmp_path):
    p = tmp_path / "per_image_shared.csv"
    _write_csv(
        p,
        [
            [0, "007", 2, 2, "background", 3, 2, 2],
            [0, "007", 2, 2, "dress", 1, 2, 1],
            [1, "008", 1, 1, "upper", 1, 1, 1],
        ],
    )
    c = results.read_image_counts(p)
    assert c.shape == (2, 3, NUM_CLASSES)
    assert c[0, :, DRESS].tolist() == [1, 2, 1] and c[1, :, BG].tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    ("rows", "msg"),
    [
        ([[0, "a", 1, 1, "shorts", 1, 1, 1]], "unknown classes"),
        ([[0, "a", 1, 1, "upper", 1, 1, 1], [2, "b", 1, 1, "upper", 1, 1, 1]], "not 0.."),
    ],
)
def test_read_image_counts_fails_on_bad_files(tmp_path, rows, msg):
    p = tmp_path / "x.csv"
    _write_csv(p, rows)
    with pytest.raises(ValueError, match=msg):
        results.read_image_counts(p)


def _image_counts(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gt = rng.integers(0, 20, size=(n, NUM_CLASSES))
    tp = (gt * rng.uniform(0.3, 1.0, size=gt.shape)).astype(np.int64)
    pred = tp + rng.integers(0, 5, size=gt.shape)
    return np.stack([gt, pred, tp], axis=1)


def test_bootstrap_is_paired_and_seeded():
    a = _image_counts(30, 0)
    s1 = results.bootstrap_miou({"A": a, "B": a.copy()}, n=40, seed=3, chunk=7)
    s2 = results.bootstrap_miou({"A": a}, n=40, seed=3, chunk=40)
    assert np.array_equal(s1["A"], s1["B"])  # same resamples for every run
    assert np.allclose(s1["A"], s2["A"])  # chunking does not change the draws' scores
    point = results.miou_from_counts(a.sum(0))
    lo, hi = results.interval(s1["A"], 0.95)
    assert lo < hi and s1["A"].min() <= point <= s1["A"].max()


def test_bootstrap_rejects_different_image_sets():
    with pytest.raises(ValueError, match="differ in shape"):
        results.bootstrap_miou({"A": _image_counts(5, 0), "B": _image_counts(6, 0)}, 2, 0)


def _metrics_from_cm(cm: np.ndarray, images: int) -> dict:
    s = scores(torch.from_numpy(cm))
    return s | {
        "images": images,
        "checkpoint_step": 8,
        "confusion_matrix": cm.tolist(),
        "classes": list(SHARED_CLASSES),
    }


def _inputs():
    cfg = {
        "runs": {
            "A": {"name": "run-a", "model": "Net A"},
            "B": {"name": "run-b", "model": "Net B"},
        },
        "datasets": ["lip"],
        "exclude": ["accessory"],
        "adjusted": {"d1": {"dataset": "lip", "cls": "dress", "raw": ["jumpsuits"]}},
        "bootstrap": {"n": 20, "seed": 0, "level": 0.9},
    }
    raw = {"A": {"lip": _raw_conf(1)}, "B": {"lip": _raw_conf(2)}}
    official = {a: {"lip": _metrics_from_cm(results.cm_from_raw(raw[a]["lip"]), 4)} for a in raw}
    meta = {a: {"params": 1_000_000, "train_seconds": 3600, "config_hash": a} for a in raw}
    counts = {a: {"lip": _image_counts(4, i)} for i, a in enumerate(raw)}
    return cfg, official, meta, raw, counts


def test_build_results_tables_and_figures(tmp_path):
    cfg, official, meta, raw, counts = _inputs()
    res = results.build_results(cfg, official, meta, raw, counts)

    assert res["crosscheck"]["A"]["lip"]["same_cm_as_official"]
    assert res["crosscheck"]["A"]["lip"]["max_abs_iou_diff_rebuilt_vs_official"] == 0
    d1 = res["adjusted"]["d1"]
    assert d1["raw_px"] == 20 and d1["share_of_class_gt"] == pytest.approx(20 / 60)
    adj = scores(torch.from_numpy(results.cm_from_raw(raw["A"]["lip"], ["jumpsuits"])))
    assert d1["runs"]["A"]["iou_adjusted"] == pytest.approx(adj["iou"]["dress"])
    assert set(res["bootstrap"]["diff"]) == {"B-A"}
    diff = res["bootstrap"]["diff"]["B-A"]["lip"]["diff"]
    assert diff == pytest.approx(
        res["official"]["B"]["lip"]["miou_wo"] - res["official"]["A"]["lip"]["miou_wo"]
    )

    md = results.tables_markdown(res)
    assert "| d1 | lip | dress | jumpsuits |" in md and "nan" not in md.lower()

    paths = [
        results.plot_headline(res, tmp_path),
        results.plot_differences(res, tmp_path),
        results.plot_per_class(res, tmp_path, "iou"),
        results.plot_confusion(official, res, "lip", tmp_path),
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    results.write_json(res, tmp_path / "r.json")


def test_build_results_rejects_adjustment_of_another_class():
    cfg, official, meta, raw, counts = _inputs()
    cfg["adjusted"]["d1"]["cls"] = "upper"
    with pytest.raises(ValueError, match="do not map to 'upper'"):
        results.build_results(cfg, official, meta, raw, counts)


def test_row_normalised_leaves_empty_rows_nan():
    r = results.row_normalised(np.array([[1, 3], [0, 0]]))
    assert r[0].tolist() == [0.25, 0.75] and np.isnan(r[1]).all()


def test_load_config_checks_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("runs: {}\n")
    with pytest.raises(ValueError, match="missing keys"):
        results.load_config(p)
