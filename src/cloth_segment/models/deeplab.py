"""DeepLabV3+ with a ResNet-50 encoder, ImageNet-pretrained."""

from __future__ import annotations

import segmentation_models_pytorch as smp
from torch import nn

from cloth_segment.data.labels import NUM_CLASSES
from cloth_segment.models import register
from cloth_segment.models.wrappers import PadToMultiple

# smp's default output stride.
# smp.DeepLabV3Plus rejects H or W not divisible by it, so PadToMultiple is needed.
OUTPUT_STRIDE = 16


@register("deeplabv3p_r50")
def build_deeplabv3p_r50(model_cfg: dict) -> nn.Module:
    """DeepLabV3+ R50 returning raw logits [B, NUM_CLASSES, H, W] for any input H×W.

    `model_cfg` keys: `name`, and `pretrained` (default true: ImageNet weights).
    The decoder uses smp defaults (256 channels, atrous rates 12/24/36).
    """
    pretrained = model_cfg.get("pretrained", True)
    net = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        encoder_output_stride=OUTPUT_STRIDE,
        classes=NUM_CLASSES,
        activation=None,
    )
    return PadToMultiple(net, OUTPUT_STRIDE)
