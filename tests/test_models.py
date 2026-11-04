"""Model registry and wrappers. CPU only, no pretrained weights, tiny inputs."""

import pytest
import torch
from torch import nn

from cloth_segment import models
from cloth_segment.data.labels import NUM_CLASSES
from cloth_segment.models import build_model, count_parameters, register
from cloth_segment.models.wrappers import PadToMultiple


class _Strided(nn.Module):
    """Toy net: needs H, W divisible by `stride`, returns logits at 1/`stride` size."""

    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(3, NUM_CLASSES, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[-2] % self.stride == 0 and x.shape[-1] % self.stride == 0
        return self.conv(x)


@pytest.fixture
def toy_model():
    name = "_toy_test_model"

    @register(name)
    def build(stride: int = 4):
        return PadToMultiple(_Strided(stride), stride)

    yield name
    del models.MODELS[name]


@pytest.mark.parametrize("shape", [(2, 3, 64, 96), (1, 3, 100, 75), (1, 3, 33, 47), (1, 3, 5, 3)])
@pytest.mark.parametrize("multiple", [1, 4, 32])
def test_pad_to_multiple_keeps_input_size(shape, multiple):
    net = PadToMultiple(_Strided(multiple), multiple)
    out = net(torch.randn(shape))
    assert out.shape == (shape[0], NUM_CLASSES, *shape[2:])
    assert out.dtype == torch.float32


def test_pad_to_multiple_is_identity_when_aligned():
    inner = nn.Conv2d(3, NUM_CLASSES, kernel_size=3, padding=1)
    x = torch.randn(1, 3, 64, 32)
    assert torch.equal(PadToMultiple(inner, 32)(x), inner(x))


def test_pad_to_multiple_crop_matches_original_pixels():
    # A per-pixel (1×1) net must give the same values with or without padding.
    inner = nn.Conv2d(3, NUM_CLASSES, kernel_size=1)
    x = torch.randn(1, 3, 33, 47)
    assert torch.allclose(PadToMultiple(inner, 32)(x), inner(x))


def test_pad_to_multiple_rejects_bad_input():
    with pytest.raises(ValueError):
        PadToMultiple(nn.Identity(), 0)
    with pytest.raises(ValueError):
        PadToMultiple(nn.Identity(), 32)(torch.randn(3, 32, 32))


def test_build_model_passes_kwargs(toy_model):
    m = build_model({"model": {"name": toy_model, "stride": 8}})
    assert m(torch.randn(1, 3, 50, 30)).shape == (1, NUM_CLASSES, 50, 30)
    assert count_parameters(m) == NUM_CLASSES * 3 * 8 * 8 + NUM_CLASSES


def test_build_model_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown model.name"):
        build_model({"model": {"name": "nope"}})
    with pytest.raises(ValueError, match="unknown model.name"):
        build_model({"model": {"name": None}})  # base.yaml default


def test_build_model_unknown_key_raises(toy_model):
    with pytest.raises(TypeError):
        build_model({"model": {"name": toy_model, "typo_key": 1}})


def test_register_rejects_duplicate(toy_model):
    with pytest.raises(ValueError, match="already registered"):
        register(toy_model)(lambda: nn.Identity())


def test_count_parameters_trainable_only():
    m = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    m[0].requires_grad_(False)
    assert count_parameters(m) == 9 + 4
    assert count_parameters(m, trainable_only=True) == 4
