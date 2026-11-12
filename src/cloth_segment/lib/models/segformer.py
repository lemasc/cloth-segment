"""SegFormer-B2 (MiT-B2 encoder + all-MLP decoder), ImageNet-pretrained.

Built with smp and set to match the original SegFormer B2 (NVlabs/SegFormer,
`local_configs/segformer/B2`) where smp's defaults differ:

- decoder width 768 (smp default 256);
- `Dropout2d` before the classifier (smp has none; the original uses
  `nn.Dropout2d(0.1)`, which drops whole channels, not single values);
- logits upsampled x4 with `align_corners=False` (smp's head uses `True`):
  smp returns 1/4-size logits (`upsampling=1`) and `PadToMultiple` upsamples
  them with `align_corners=False`.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
from torch import nn

from cloth_segment.lib.data.labels import NUM_CLASSES
from cloth_segment.lib.models import register
from cloth_segment.lib.models.wrappers import PadToMultiple

# MiT downsamples 5 times in smp's encoder contract, so smp.Segformer rejects
# H or W not divisible by 32.
OUTPUT_STRIDE = 32
DECODER_CHANNELS = 768  # B2 decoder width in the paper
DEFAULT_CLASSIFIER_DROPOUT = 0.1  # original B2 config: dropout_ratio=0.1


@register("segformer_b2")
def build_segformer_b2(model_cfg: dict) -> nn.Module:
    """SegFormer-B2 returning raw logits [B, NUM_CLASSES, H, W] for any input H×W.

    `model_cfg` keys: `name`, `pretrained` (default true: ImageNet weights) and
    `classifier_dropout` (default 0.1: `Dropout2d` rate before the classifier;
    0 removes it).
    """
    pretrained = model_cfg.get("pretrained", True)
    dropout = float(model_cfg.get("classifier_dropout", DEFAULT_CLASSIFIER_DROPOUT))
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"classifier_dropout must be in [0, 1) (got {dropout})")
    net = smp.Segformer(
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        decoder_segmentation_channels=DECODER_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,
        upsampling=1,  # return 1/4-size logits; PadToMultiple upsamples them
    )
    # Before the 1×1 classifier conv, after the decoder's fuse conv + BN + ReLU.
    # Always insert a module (Identity when 0) so the classifier's state_dict key
    # stays `segmentation_head.1` and checkpoints load whatever the dropout rate.
    net.segmentation_head.insert(0, nn.Dropout2d(dropout) if dropout > 0 else nn.Identity())
    return PadToMultiple(net, OUTPUT_STRIDE)
