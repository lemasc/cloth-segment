"""Model A (T6): U-Net with a ResNet-34 encoder, ImageNet-pretrained."""

from __future__ import annotations

import segmentation_models_pytorch as smp
from torch import nn

from cloth_segment.data.labels import NUM_CLASSES
from cloth_segment.models import register
from cloth_segment.models.wrappers import PadToMultiple

# The encoder downsamples 5 times, so smp.Unet needs H and W divisible by 32.
OUTPUT_STRIDE = 32


@register("unet_r34")
def build_unet_r34(pretrained: bool = True) -> nn.Module:
    """U-Net R34 returning raw logits [B, NUM_CLASSES, H, W] for any input H×W."""
    net = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        classes=NUM_CLASSES,
        activation=None,
    )
    return PadToMultiple(net, OUTPUT_STRIDE)
