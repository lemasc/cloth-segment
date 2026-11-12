"""Model registry and wrappers. CPU only, no pretrained weights, tiny inputs."""

import pytest
import torch
import torch.nn.functional as F
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
    assert set(cfg) == {"model"}  # an experiment config overrides model.* only
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
    assert set(cfg) == {"model"}  # an experiment config overrides model.* only
    assert cfg["model"]["name"] == "deeplabv3p_r50"
    assert cfg["model"]["pretrained"] is True
    cfg["model"]["pretrained"] = False  # no download in tests
    assert isinstance(build_model(cfg["model"]), nn.Module)


@pytest.fixture(scope="module")
def segformer():
    m = build_model({"name": "segformer_b2", "pretrained": False})
    return m.eval()


# (1, 3, 100, 75) and (1, 3, 33, 47) are not divisible by 32: bare smp.Segformer rejects them.
@pytest.mark.parametrize("shape", [(2, 3, 64, 96), (1, 3, 100, 75), (1, 3, 33, 47)])
def test_segformer_b2_output_shape(segformer, shape):
    with torch.no_grad():
        out = segformer(torch.randn(shape))
    assert out.shape == (shape[0], NUM_CLASSES, *shape[2:])
    assert out.dtype == torch.float32


def test_segformer_b2_returns_logits_not_probabilities(segformer):
    with torch.no_grad():
        out = segformer(torch.randn(1, 3, 64, 64))
    assert not torch.allclose(out.softmax(1).sum(1), out.sum(1))


def test_segformer_b2_architecture(segformer):
    from cloth_segment.models.segformer import OUTPUT_STRIDE

    assert isinstance(segformer, PadToMultiple)
    assert segformer.multiple == OUTPUT_STRIDE == 32
    net = segformer.net
    # MiT-B2, not B0/B1: embed dims and depths from the paper.
    assert net.encoder.out_channels[2:] == [64, 128, 320, 512]
    assert [len(b) for b in (net.encoder.block1, net.encoder.block2)] == [3, 4]
    assert [len(b) for b in (net.encoder.block3, net.encoder.block4)] == [6, 3]
    # B2 decoder width 768, not smp's default 256.
    assert net.decoder.fuse_stage[0].out_channels == 768
    # Head: Dropout2d(0.1) -> 1x1 classifier, and no x4 upsampling inside smp.
    head = list(net.segmentation_head)
    assert isinstance(head[0], nn.Dropout2d) and head[0].p == 0.1
    assert isinstance(head[1], nn.Conv2d) and head[1].kernel_size == (1, 1)
    assert not any(isinstance(m, nn.Upsample) for m in net.modules())


def test_segformer_b2_upsamples_with_align_corners_false(segformer):
    # smp returns 1/4-size logits; the wrapper pads to 32, upsamples x4 with
    # align_corners=False (as in the original SegFormer) and crops.
    x = torch.randn(1, 3, 64, 50)
    with torch.no_grad():
        low = segformer.net(F.pad(x, (0, 14)))
        out = segformer(x)
    assert low.shape == (1, NUM_CLASSES, 16, 16)
    ref = F.interpolate(low, size=(64, 64), mode="bilinear", align_corners=False)[..., :50]
    assert torch.equal(out, ref)


def test_segformer_b2_dropout_changes_nothing_but_training():
    def build(cfg):
        torch.manual_seed(0)
        return build_model({"name": "segformer_b2", "pretrained": False, **cfg}).eval()

    with_drop, without = build({}), build({"classifier_dropout": 0})
    assert not any(isinstance(m, nn.Dropout2d) for m in without.modules())
    # Dropout has no weights: same seed, same initial weights and same eval output.
    # Same keys too, so a checkpoint loads whatever the dropout rate.
    a, b = with_drop.state_dict(), without.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(with_drop(x), without(x))


@pytest.mark.parametrize("p", [-0.1, 1.0])
def test_segformer_b2_rejects_bad_dropout(p):
    with pytest.raises(ValueError, match="classifier_dropout"):
        build_model({"name": "segformer_b2", "pretrained": False, "classifier_dropout": p})


def test_segformer_b2_trains_one_step_in_train_mode():
    m = build_model({"name": "segformer_b2", "pretrained": False}).train()
    out = m(torch.randn(2, 3, 64, 64))
    out.sum().backward()
    assert out.shape == (2, NUM_CLASSES, 64, 64)
    assert m.net.segmentation_head[1].weight.grad is not None


def test_segformer_b2_config():
    cfg = yaml.safe_load((paths.CONFIGS / "segformer_b2.yaml").read_text())
    assert set(cfg) == {"model"}  # an experiment config overrides model.* only
    assert cfg["model"]["name"] == "segformer_b2"
    assert cfg["model"]["pretrained"] is True
    assert cfg["model"]["classifier_dropout"] == 0.1
    cfg["model"]["pretrained"] = False  # no download in tests
    assert isinstance(build_model(cfg["model"]), nn.Module)
