"""T5 test-set entry point: run lookup, metrics.json content, no overwrite. CPU only."""

import datetime as dt
import json

import pytest
import torch
import yaml
from test_dataset import _write_fixture
from torch import nn

from cloth_segment import evaluate, models, paths, runs
from cloth_segment.data.dataset import SegDataset, class_pixel_counts
from cloth_segment.data.labels import NUM_CLASSES, SHARED_CLASSES

SPECS = [
    ("lip", "test", 120, 70, False),
    ("lip", "test", 90, 64, True),
    ("cihp", "test", 100, 150, False),
]


class _Constant(nn.Module):
    """Predicts one class everywhere at 1/4 resolution; the class is a saved parameter."""

    def __init__(self, cfg):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(NUM_CLASSES))

    def forward(self, x):
        b, _, h, w = x.shape
        return self.bias.view(1, -1, 1, 1).expand(b, -1, h // 4, w // 4)


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "REGISTRY", {"const": _Constant})
    cfg = yaml.safe_load((paths.CONFIGS / "base.yaml").read_text())
    cfg["model"]["name"] = "const"
    cfg["input"]["size"] = 64
    cfg["train"]["amp"] = "none"
    rd = runs.create_run(
        cfg, "t", tmp_path / "runs", now=dt.datetime(2026, 11, 5), git=("abc1234", False)
    )
    (rd / "ckpt").mkdir()
    model = _Constant(cfg)
    with torch.no_grad():
        model.bias[4] = 1.0
    torch.save({"step": 123, "model": model.state_dict()}, rd / "ckpt" / "best.pt")
    runs.mark_finished(rd)
    return rd


def _maker(tmp_path):
    df, ir, mr = _write_fixture(tmp_path / "fx", SPECS)

    def make(cfg, test_set):
        rows = df[df.dataset == test_set]
        return SegDataset(rows, cfg["input"]["size"], train=False, image_root=ir, mask_root=mr)

    return df, make


def test_evaluate_run_writes_metrics(tmp_path, run_dir):
    df, make = _maker(tmp_path)
    out = evaluate.evaluate_run(
        run_dir, ["lip", "cihp"], torch.device("cpu"), num_workers=0, make_dataset=make
    )
    for test_set in ("lip", "cihp"):
        path = out[test_set]
        assert path == run_dir / "eval" / evaluate.EVAL_VERSION / test_set / "metrics.json"
        m = json.loads(path.read_text())
        rows = df[df.dataset == test_set]
        cm = torch.tensor(m["confusion_matrix"])
        assert cm[:, 4].tolist() == class_pixel_counts(rows).tolist()  # all predicted class 4
        assert cm.sum() == cm[:, 4].sum()
        assert m["images"] == len(rows) and m["checkpoint_step"] == 123
        assert m["classes"] == list(SHARED_CLASSES) and m["test_set"] == test_set
        assert m["config_hash"] == runs.read_meta(run_dir)["config_hash"]
        assert m["eval_version"] == evaluate.EVAL_VERSION


def test_evaluate_run_never_overwrites(tmp_path, run_dir):
    _, make = _maker(tmp_path)
    cpu = torch.device("cpu")
    path = evaluate.evaluate_run(run_dir, ["lip"], cpu, num_workers=0, make_dataset=make)["lip"]
    before = path.read_text()

    def boom(cfg, test_set):
        raise AssertionError("cached test set must not be rebuilt")

    evaluate.evaluate_run(run_dir, ["lip"], cpu, num_workers=0, make_dataset=boom)
    assert path.read_text() == before


def test_resolve_run(tmp_path, run_dir):
    assert evaluate.resolve_run(run_dir) == run_dir
    assert evaluate.resolve_run(run_dir.name, runs_dir=run_dir.parent) == run_dir
    with pytest.raises(FileNotFoundError):
        evaluate.resolve_run("nope", runs_dir=tmp_path)
    runs.update_meta(run_dir, status="failed")
    with pytest.raises(ValueError, match="not finished"):
        evaluate.resolve_run(run_dir)


def test_load_test_set_rejects_unknown():
    with pytest.raises(ValueError, match="test set"):
        evaluate.load_test_set({"data": {}}, "atr")


def test_main_rejects_unknown_test_set():
    with pytest.raises(SystemExit):
        evaluate.main(["some-run", "atr"])
