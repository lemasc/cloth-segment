import pytest
import torch
from torch import nn

from cloth_segment.data.labels import NUM_CLASSES
from cloth_segment.probe import (
    ARCHITECTURES,
    ProbeResult,
    format_table,
    probe_models,
    probe_step,
)

CPU = torch.device("cpu")


class _Downsampling(nn.Module):
    """Predicts at 1/4 resolution, like SegFormer."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, NUM_CLASSES, kernel_size=4, stride=4)

    def forward(self, x):
        return self.conv(x)


class _OOM(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        raise torch.OutOfMemoryError("CUDA out of memory. fake")


def test_probe_step_cpu_upsamples_logits():
    r = probe_step("tiny", _Downsampling(), 2, size=32, device=CPU, amp=True, iters=2, warmup=1)
    assert r.ok
    assert r.peak_mem_mib is None  # no GPU
    assert r.sec_per_iter is not None and r.sec_per_iter > 0


def test_probe_step_reports_oom():
    r = probe_step("oom", _OOM(), 4, size=32, device=CPU, amp=False, iters=1, warmup=0)
    assert not r.ok
    assert r.sec_per_iter is None
    assert "out of memory" in r.error


def test_probe_models_stops_after_oom():
    calls = []

    def build_oom():
        calls.append(1)
        return _OOM()

    res = probe_models(
        ["oom"], [8, 2, 4], size=32, device=CPU, iters=1, warmup=0, builders={"oom": build_oom}
    )
    assert [r.batch_size for r in res] == [2]  # sorted, then stop at first OOM
    assert len(calls) == 1


def test_format_table():
    rows = [
        ProbeResult("a", 8, True, 6000.0, 0.5),
        ProbeResult("a", 16, False, None, None, "oom"),
    ]
    table = format_table(rows, total_mib=12000.0)
    assert "| a | 8 | yes | 6000 | 50 | 0.500 | 16.0 |" in table
    assert "| a | 16 | OOM | - | - | - | - |" in table


@pytest.mark.parametrize(
    ("name", "stride", "params_m"),
    [
        ("unet_r34", 1, (20, 30)),
        ("deeplabv3p_r50", 1, (22, 30)),
        ("segformer_b2", 4, (25, 30)),
    ],
)
def test_architectures_build(name, stride, params_m):
    model = ARCHITECTURES[name]().eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, NUM_CLASSES, 64 // stride, 64 // stride)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    assert params_m[0] < n < params_m[1], n


def test_probe_models_over_cap_counts_as_failure(monkeypatch):
    import cloth_segment.probe as probe

    seen = []
    monkeypatch.setattr(probe, "_peak", lambda device: 5000.0)
    res = probe_models(
        ["tiny"],
        [2, 4],
        size=16,
        device=CPU,
        iters=1,
        warmup=0,
        builders={"tiny": _Downsampling},
        max_mib=1000.0,
        on_result=seen.append,
    )
    assert [r.batch_size for r in res] == [2] and seen == res
    assert not res[0].ok and "> 1000" in res[0].error
    assert "| tiny | 2 | over |" in format_table(res)
