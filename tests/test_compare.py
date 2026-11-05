"""Run comparison from files: readers, mIoU without a class, tables, plots. No data, no GPU."""

import json

import numpy as np
import pytest

from cloth_segment import compare

CLASSES = ["background", "upper", "accessory"]
DS = ("lip", "cihp")


def _metrics(iou: list[float], cm: list[list[int]], step: int = 8, version: str = "eval-v2"):
    gt = [sum(r) for r in cm]
    return {
        "images": 3,
        "checkpoint_step": step,
        "checkpoint_select": "dataset_mean_miou",
        "eval_version": version,
        "miou": float(np.mean(iou)),
        "mdice": 0.5,
        "pixel_acc": 0.9,
        "iou": dict(zip(CLASSES, iou)),
        "gt_pixels": dict(zip(CLASSES, gt)),
        "in_mean": CLASSES,
        "classes": CLASSES,
        "confusion_matrix": cm,
    }


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _run(tmp_path, name, per_dataset_log: bool, test_version: str = "eval-v2"):
    """A finished run with 8 steps, val every 4, test metrics under `test_version`."""
    d = tmp_path / name
    _write(d / "meta.json", {"status": "finished", "steps": 8, "best_step": 8, "commit": "abc"})
    log = [{"step": s, "loss": 1.0 / s, "lr": 0.1} for s in (2, 4, 6, 8)]
    for s, v in ((4, 0.4), (8, 0.6)):
        if per_dataset_log:
            for k, ds in enumerate(DS):
                iou = {"background": 0.9, "upper": v, "accessory": 0.0}
                log.append({"step": s, "split": "val", "dataset": ds, "miou": v + k, "iou": iou})
            log.append({"step": s, "split": "val", "select": "dataset_mean_miou", "value": v})
        else:
            log.append({"step": s, "split": "val", "miou": v, "iou": {"accessory": 0.0}})
    (d / "log.jsonl").write_text("\n".join(json.dumps(x) for x in log) + "\n\n")
    cm = [[90, 5, 5], [0, 8, 2], [0, 3, 1]]
    for ds in DS:
        _write(d / "eval" / test_version / ds / "metrics.json", _metrics([0.9, 0.6, 0.1], cm))
    return d


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert compare.read_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_miou_without_drops_only_excluded_classes():
    m = _metrics([0.9, 0.6, 0.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert compare.miou_without(m) == pytest.approx(0.5)
    assert compare.miou_without(m, ["accessory"]) == pytest.approx(0.75)
    m["in_mean"] = ["background", "upper"]  # a class not in the mean stays out
    assert compare.miou_without(m) == pytest.approx(0.75)
    assert compare.miou_without(m, CLASSES) is None


def test_test_metrics_uses_newest_complete_version(tmp_path):
    d = _run(tmp_path, "r", True, test_version="eval-v2")
    # an eval-v10 folder with only one dataset is incomplete and skipped
    _write(d / "eval" / "eval-v10" / "lip" / "metrics.json", {"eval_version": "eval-v10"})
    _write(d / "eval" / "eval-v1" / "lip" / "metrics.json", {"eval_version": "eval-v1"})
    got = compare.test_metrics(d, DS)
    assert {m["eval_version"] for m in got.values()} == {"eval-v2"}
    with pytest.raises(FileNotFoundError):
        compare.test_metrics(tmp_path / "missing", DS)


def test_val_at_step_falls_back_to_val_metrics_files(tmp_path):
    new = compare.load_run(_run(tmp_path, "new", True), DS)
    assert compare.val_at_step(new, 8, DS)["cihp"]["miou"] == pytest.approx(1.6)

    d = _run(tmp_path, "old", False, test_version="eval-v1")
    m = _metrics([0.9, 0.5, 0.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], step=8)
    _write(d / "eval" / "eval-v2" / "val" / "lip" / "metrics.json", m)
    old = compare.load_run(d, DS)
    got = compare.val_at_step(old, 8, DS)
    assert set(got) == {"lip"}  # no cihp val file, and the pooled log has no dataset
    assert compare.val_at_step(old, 4, DS) == {}


def test_loss_blocks_and_predicted_as():
    log = [{"step": s, "loss": float(s), "lr": 0.1} for s in (1, 2, 3, 4)] + [{"step": 4, "x": 1}]
    assert compare.loss_blocks(log, 2) == [(0, 1.5, 2), (2, 3.5, 2)]
    m = _metrics([0.9, 0.6, 0.1], [[90, 5, 5], [0, 8, 2], [0, 3, 1]])
    p = compare.predicted_as(m, "accessory")
    assert (p["gt"], p["predicted"], p["correct"]) == (4, 8, 1)
    assert p["gt_share"]["upper"] == pytest.approx(0.75)


def test_comparison_markdown_has_values_from_files(tmp_path):
    old = compare.load_run(_run(tmp_path, "old", False, test_version="eval-v1"), DS)
    new = compare.load_run(_run(tmp_path, "new", True), DS)
    md = compare.comparison_markdown(old, new, exclude=["accessory"], block=4, datasets=DS)
    assert "| params | (not in meta) | (not in meta) |" in md
    assert "| eval_version (test metrics.json) | eval-v2 | eval-v2 |" in md
    # all classes (0.9+0.6+0.1)/3 and without accessory (0.9+0.6)/2
    assert "| new | 0.5333 | 0.7500 |" in md
    assert "| new - old | +0.0000 | +0.0000 |" in md
    assert "| lip | 3.509% | new | 8 | 1 | upper 75.0%, accessory 25.0% |" in md
    assert "- select value: 4→8 +0.2000" in md
    assert "| 4–8 |" in md


def test_plots_write_pngs_only_into_out(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    for name, per_ds in (("old", False), ("new", True)):
        run = compare.load_run(_run(tmp_path, name, per_ds), DS)
        for path in (
            compare.plot_curves(run, out, shade_last=4, datasets=DS),
            compare.plot_lr(run, out),
        ):
            assert path.parent == out and path.read_bytes()[:4] == b"\x89PNG"
    assert sorted(p.name for p in out.iterdir()) == [
        "curves_new.png",
        "curves_old.png",
        "lr_new.png",
        "lr_old.png",
    ]
