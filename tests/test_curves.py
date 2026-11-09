import json

import pytest

from cloth_segment import curves

DS = ["lip", "cihp", "dfmm"]


def _log(with_select: bool = True) -> list[dict]:
    log = []
    for step in range(50, 201, 50):
        log.append(
            {"step": step, "loss": 1.0 if step <= 100 else 0.5, "lr": 1e-4 * (1 - step / 200)}
        )
    for step, base in ((100, 0.4), (200, 0.5)):
        for i, d in enumerate(DS):
            log.append({"step": step, "split": "val", "dataset": d, "miou": base + 0.1 * i})
        log.append({"step": step, "split": "val", "dataset": "pooled", "miou": 0.9})
        if with_select:
            log.append(
                {"step": step, "split": "val", "select": "dataset_mean_miou", "value": base + 0.1}
            )
    return log


def test_run_rows_values_from_log():
    rows = curves.run_rows(_log(), DS)
    assert [r["step"] for r in rows] == [100, 200]
    assert rows[0]["val"] == pytest.approx({"lip": 0.4, "cihp": 0.5, "dfmm": 0.6})
    assert rows[1]["mean"] == 0.6  # the logged selection value
    assert rows[0]["loss"] == 1.0 and rows[1]["loss"] == 0.5  # mean since previous check
    assert rows[1]["lr"] == 0.0


def test_run_rows_mean_falls_back_to_datasets():
    rows = curves.run_rows(_log(with_select=False), DS)
    assert abs(rows[0]["mean"] - 0.5) < 1e-12


def test_summarise_best_last_and_gains():
    s = curves.summarise(curves.run_rows(_log(), DS))
    assert s["best_step"] == 200 and s["last_step"] == 200
    assert len(s["gains"]) == 1 and abs(s["gains"][0]["gain"] - 0.1) < 1e-12


def test_summarise_empty():
    assert curves.summarise([])["best_step"] is None


def test_build_markdown_json_and_plot(tmp_path):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "log.jsonl").write_text("\n".join(json.dumps(x) for x in _log()) + "\n")
    res = curves.build({"X9": {"name": "r1", "model": "M"}}, tmp_path / "runs", DS)
    md = curves.markdown(res, DS)
    assert "| X9 | M | 200 | 200 | 0.6000 | 100→200 +0.1000 | 0.5000 |" in md
    assert json.loads(curves.write_json(res, tmp_path / "c.json").read_text())["X9"]["summary"]
    png = curves.plot(res, tmp_path / "fig" / "c.png")
    assert png.exists() and png.stat().st_size > 0
