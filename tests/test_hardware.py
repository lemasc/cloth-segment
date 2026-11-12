"""Hardware fields and backfill: torch.cuda and nvidia-smi are mocked, no GPU needed."""

import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from cloth_segment.lib import hardware
from cloth_segment.lib.hardware import BackfillRefused, backfill_run, driver_version, gpu_fields

HOST = "box-a"
NEW = {
    "gpu_name": "Fake RTX",
    "gpu_total_mib": 8192.0,
    "gpu_capability": "8.6",
    "driver_version": "550.54",
}


@pytest.fixture
def fake_cuda(monkeypatch):
    """One fake GPU with a working nvidia-smi."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Fake RTX")
    props = SimpleNamespace(total_memory=8192 * 2**20, major=8, minor=6)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)

    def smi(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="550.54\n550.54\n")

    monkeypatch.setattr(hardware.subprocess, "run", smi)


def _smi_fails(monkeypatch, exc):
    def boom(*a, **kw):
        raise exc

    monkeypatch.setattr(hardware.subprocess, "run", boom)


def _run(tmp_path, **meta):
    d = tmp_path / "20261106-0000-abc1234-x"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"host": HOST, "device": "cuda", **meta}))
    return d


def _meta(d):
    return json.loads((d / "meta.json").read_text())


def test_cuda_fields(fake_cuda):
    assert gpu_fields(torch.device("cuda")) == NEW
    assert gpu_fields(torch.device("cuda:0")) == NEW


def test_cpu_gives_nulls(fake_cuda):
    out = gpu_fields(torch.device("cpu"))
    assert out == dict.fromkeys(hardware.GPU_FIELDS)


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("nvidia-smi"),
        subprocess.CalledProcessError(9, "nvidia-smi"),
        subprocess.TimeoutExpired("nvidia-smi", 30),
    ],
)
def test_nvidia_smi_failure_gives_null_driver(fake_cuda, monkeypatch, exc):
    _smi_fails(monkeypatch, exc)
    assert driver_version() is None
    out = gpu_fields(torch.device("cuda"))
    assert out["driver_version"] is None
    assert out["gpu_name"] == "Fake RTX"  # the rest is still recorded


def test_backfill_fills_missing_only(fake_cuda, tmp_path):
    d = _run(tmp_path, gpu_name="Old Name", status="finished")
    changes = backfill_run(d, hostname=HOST)
    assert set(changes) == set(NEW) - {"gpu_name"}
    meta = _meta(d)
    assert meta["gpu_name"] == "Old Name"  # never overwritten
    assert meta["driver_version"] == "550.54"
    assert meta["hw_backfilled"] is True
    assert meta["status"] == "finished"


def test_backfill_does_not_overwrite_null(fake_cuda, tmp_path):
    d = _run(tmp_path, **dict.fromkeys(NEW))  # a recorded null is a value, not missing
    before = _meta(d)
    assert backfill_run(d, hostname=HOST) == {}
    assert _meta(d) == before  # no flag either


def test_backfill_host_mismatch_refused(fake_cuda, tmp_path):
    d = _run(tmp_path, host="box-b")
    before = (d / "meta.json").read_text()
    with pytest.raises(BackfillRefused, match="box-b"):
        backfill_run(d, hostname=HOST)
    assert (d / "meta.json").read_text() == before


def test_backfill_dry_run_writes_nothing(fake_cuda, tmp_path):
    d = _run(tmp_path)
    before = (d / "meta.json").read_text()
    assert backfill_run(d, dry_run=True, hostname=HOST) == NEW
    assert (d / "meta.json").read_text() == before


def test_backfill_cpu_run_gets_nulls(tmp_path):
    d = _run(tmp_path, device="cpu")
    assert backfill_run(d, hostname=HOST) == dict.fromkeys(NEW)
    assert _meta(d)["gpu_name"] is None
