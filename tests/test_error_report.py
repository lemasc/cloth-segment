"""T10 report from files: zone summaries, size buckets, review tally, tables, figures.
Tiny synthetic inputs; no data, no GPU."""

import json

import numpy as np
import pandas as pd
import pytest

from cloth_segment import error_report, errors, results
from cloth_segment.data.labels import NUM_CLASSES, SHARED_CLASSES

BG, UPPER, OUTER, PANTS = (
    SHARED_CLASSES.index(c) for c in ("background", "upper", "outer", "pants")
)


def _zone_json(tmp_path, name="z.json"):
    gt = np.full((20, 20), BG, dtype=np.uint8)
    gt[2:10, 4:16] = UPPER
    gt[10:18, 4:16] = PANTS
    pred = gt.copy()
    pred[gt == UPPER] = OUTER
    pred[13, 9] = BG
    zc = errors.zone_confusions(pred, gt, [0.02, 0.05])
    p = tmp_path / name
    p.write_text(
        json.dumps(
            {
                "images": 1,
                "fracs": [0.02, 0.05],
                "zones": list(errors.ZONES),
                "zone_confusion": zc.tolist(),
                "check_vs_saved_metrics": {"same_matrix": True, "max_abs_diff": 0},
            }
        )
    )
    return p


def test_zone_summary_shares_swaps_and_no_edge_score(tmp_path):
    z = error_report.read_zones(_zone_json(tmp_path))
    s = error_report.zone_summary(z, 0.02, ["accessory"])
    assert s["wrong_px"] == 8 * 12 + 1
    assert sum(s["shares"]["0.02"].values()) == pytest.approx(1.0)
    assert s["top_swaps"][0]["gt"] == "upper" and s["top_swaps"][0]["pred"] == "outer"
    assert s["class_shares"]["pants"]["inner"] > 0
    with pytest.raises(ValueError, match="headline"):
        error_report.zone_summary(z, 0.03, [])


def test_read_zones_rejects_other_zone_names(tmp_path):
    p = _zone_json(tmp_path)
    d = json.loads(p.read_text())
    d["zones"] = ["a", "b", "c"]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="zones"):
        error_report.read_zones(p)


def _counts(n: int) -> np.ndarray:
    c = np.zeros((n, 3, NUM_CLASSES), dtype=np.int64)
    c[:, 0, BG] = 80
    c[:, 0, UPPER] = 20
    c[:, 1] = c[:, 0]
    c[:, 2] = c[:, 0]
    c[0, 2, UPPER] = 10  # image 0 gets half of its upper wrong
    c[0, 1, UPPER] = 10
    return c


def test_person_share_and_bucket_scores():
    c = _counts(4)
    assert error_report.person_share(c) == pytest.approx([0.2] * 4)
    vals = np.array([10.0, 10.0, 300.0, 300.0])
    out = error_report.bucket_scores(c, vals, [0, 100, 1000], ["small", "big"], ["accessory"])
    assert [r["images"] for r in out] == [2, 2]
    assert out[1]["miou_wo"] == pytest.approx(1.0)
    assert out[0]["miou_wo"] == pytest.approx(
        results.miou_from_counts(c[:2].sum(0), results.keep_mask(["accessory"]))
    )
    empty = error_report.bucket_scores(c, vals, [0, 5, 1000, 2000], ["a", "b", "c"], [])
    assert empty[0]["images"] == 0 and empty[0]["miou_wo"] is None
    with pytest.raises(ValueError):
        error_report.bucket_scores(c, vals, [0, 1], ["a", "b"], [])


def test_review_summary_counts_per_pick():
    samples = pd.DataFrame(
        {"key": ["cihp:1", "cihp:2", "cihp:3"], "pick": ["worst", "worst", "random"]}
    )
    verdicts = pd.DataFrame(
        {
            "question": ["q", "q", "d5"],
            "key": ["cihp:1", "cihp:3", "limitation"],
            "verdict": ["W", "M", "x"],
            "note": ["", "", ""],
        }
    )
    r = error_report.review_summary(samples, verdicts, {"W": "wrong", "M": "model"})
    assert r["reviewed"] == 2 and r["images"] == 3
    assert r["counts"]["W"] == {"worst": 1, "random": 0}
    assert r["counts"]["M"] == {"worst": 0, "random": 1}


def test_build_tables_and_figures(tmp_path):
    z = error_report.read_zones(_zone_json(tmp_path))
    cfg = {
        "runs": {"A": {"name": "a", "model": "Net A"}, "B": {"name": "b", "model": "Net B"}},
        "datasets": ["lip"],
        "exclude": ["accessory"],
        "zones": {"fracs": [0.02, 0.05], "headline": 0.02},
        "buckets": {
            "short_side": {
                "title": "short side",
                "value": "short_side",
                "datasets": ["lip"],
                "edges": [0, 100, 1000],
                "labels": ["small", "big"],
            },
            "person_share": {
                "title": "person",
                "value": "person_share",
                "datasets": ["lip"],
                "edges": [0, 0.5, 1.01],
                "labels": ["<50%", ">=50%"],
            },
        },
    }
    counts = {a: {"lip": _counts(4)} for a in ("A", "B")}
    sizes = {"lip": pd.DataFrame({"h": [50, 60, 400, 500], "w": [40, 50, 300, 300]})}
    res = error_report.build(cfg, {"A": {"lip": z}, "B": {"lip": z}}, counts, sizes, None)
    assert res["buckets"]["short_side"]["per_dataset"]["lip"]["A"][0]["images"] == 2
    md = error_report.tables_markdown(res)
    assert "upper → outer" in md and "nan" not in md.lower()
    paths = [
        error_report.plot_zone_shares(res, tmp_path),
        error_report.plot_class_zones(res, tmp_path),
        error_report.plot_buckets(res, "short_side", tmp_path),
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    cfg["buckets"]["short_side"]["value"] = "colour"
    with pytest.raises(ValueError, match="unknown bucket value"):
        error_report.build(cfg, {"A": {"lip": z}, "B": {"lip": z}}, counts, sizes, None)


def test_agreement_counts_same_verdicts_per_pick():
    samples = pd.DataFrame({"key": ["c:1", "c:2", "c:3"], "pick": ["worst", "random", "random"]})
    a = pd.DataFrame({"key": ["c:1", "c:2", "c:3"], "verdict": ["M", "W", "G"]})
    b = pd.DataFrame({"key": ["c:1", "c:2"], "verdict": ["A", "W"]})
    ag = error_report.agreement(samples, a, b)
    assert ag["judged_by_both"] == 2 and ag["same"] == 1
    assert ag["per_pick"] == {"worst": 0, "random": 1}
    assert ag["differ"] == [{"key": "c:1", "pick": "worst", "a": "M", "b": "A"}]


def test_unanimous_miss_needs_every_run_below():
    a = pd.DataFrame({"index": [0, 1, 2], "raw_px": [10, 30, 60], "recall": [0.0, 0.05, 0.9]})
    b = pd.DataFrame({"index": [0, 1, 2], "raw_px": [10, 30, 60], "recall": [0.0, 0.5, 0.0]})
    u = error_report.unanimous_miss({"A": a, "B": b}, 0.1)
    assert u["missed_images"] == 1 and u["missed_share_of_pixels"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        error_report.unanimous_miss({"A": a, "B": b.iloc[:2]}, 0.1)
