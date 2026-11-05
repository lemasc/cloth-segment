"""Model registry and wrappers. CPU only, no pretrained weights, tiny inputs."""

import pytest
import torch
import yaml
from torch import nn

from cloth_segment import models, paths
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
    def build(model_cfg: dict):
        stride = model_cfg.get("stride", 4)
        return PadToMultiple(_Strided(stride), stride)

    yield name
    del models.REGISTRY[name]


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
    m = build_model({"name": toy_model, "stride": 8})
    assert m(torch.randn(1, 3, 50, 30)).shape == (1, NUM_CLASSES, 50, 30)
    assert count_parameters(m) == NUM_CLASSES * 3 * 8 * 8 + NUM_CLASSES


def test_register_rejects_duplicate(toy_model):
    with pytest.raises(ValueError, match="registered twice"):
        register(toy_model)(lambda cfg: nn.Identity())


@pytest.fixture(scope="module")
def unet():
    m = build_model({"name": "unet_r34", "pretrained": False})
    return m.eval()


@pytest.mark.parametrize("shape", [(2, 3, 64, 96), (1, 3, 100, 75), (1, 3, 33, 47)])
def test_unet_r34_output_shape(unet, shape):
    with torch.no_grad():
        out = unet(torch.randn(shape))
    assert out.shape == (shape[0], NUM_CLASSES, *shape[2:])
    assert out.dtype == torch.float32


def test_unet_r34_returns_logits_not_probabilities(unet):
    with torch.no_grad():
        out = unet(torch.randn(1, 3, 64, 64))
    assert not torch.allclose(out.softmax(1).sum(1), out.sum(1))


def test_unet_r34_config():
    cfg = yaml.safe_load((paths.CONFIGS / "unet_r34.yaml").read_text())
    assert set(cfg) == {"model"}  # D5: an experiment config overrides model.* only
    assert cfg["model"]["name"] == "unet_r34"
    assert cfg["model"]["pretrained"] is True
    cfg["model"]["pretrained"] = False  # no download in tests
    assert isinstance(build_model(cfg["model"]), nn.Module)


def test_count_parameters_trainable_only():
    m = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    m[0].requires_grad_(False)
    assert count_parameters(m) == 9 + 4
    assert count_parameters(m, trainable_only=True) == 4


@pytest.fixture(scope="module")
def deeplab():
    m = build_model({"name": "deeplabv3p_r50", "pretrained": False})
    return m.eval()


# (1, 3, 100, 75) is not divisible by 16: bare smp.DeepLabV3Plus rejects it.
@pytest.mark.parametrize("shape", [(2, 3, 64, 96), (1, 3, 100, 75), (1, 3, 33, 47)])
def test_deeplabv3p_r50_output_shape(deeplab, shape):
    with torch.no_grad():
        out = deeplab(torch.randn(shape))
    assert out.shape == (shape[0], NUM_CLASSES, *shape[2:])
    assert out.dtype == torch.float32


def test_deeplabv3p_r50_returns_logits_not_probabilities(deeplab):
    with torch.no_grad():
        out = deeplab(torch.randn(1, 3, 64, 64))
    assert not torch.allclose(out.softmax(1).sum(1), out.sum(1))


def test_deeplabv3p_r50_architecture(deeplab):
    from cloth_segment.models.deeplab import OUTPUT_STRIDE

    assert isinstance(deeplab, PadToMultiple)
    assert deeplab.multiple == OUTPUT_STRIDE == 16
    assert deeplab.net.encoder.output_stride == 16
    # ResNet-50, not ResNet-34: layer1 uses bottleneck blocks (4x channel expansion).
    assert deeplab.net.encoder.layer1[0].conv3.out_channels == 256


def test_deeplabv3p_r50_trains_one_step_in_train_mode():
    # Batch norm in the ASPP pooling branch fails on batch 1 (why train uses drop_last).
    m = build_model({"name": "deeplabv3p_r50", "pretrained": False}).train()
    out = m(torch.randn(2, 3, 64, 64))
    out.sum().backward()
    assert out.shape == (2, NUM_CLASSES, 64, 64)


def test_deeplabv3p_r50_config():
    cfg = yaml.safe_load((paths.CONFIGS / "deeplabv3p_r50.yaml").read_text())
    assert set(cfg) == {"model"}  # D5: an experiment config overrides model.* only
    assert cfg["model"]["name"] == "deeplabv3p_r50"
    assert cfg["model"]["pretrained"] is True
    cfg["model"]["pretrained"] = False  # no download in tests
    assert isinstance(build_model(cfg["model"]), nn.Module)
