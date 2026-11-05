"""Wrappers that make any segmentation network return logits at the input size."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PadToMultiple(nn.Module):
    """Pad H and W up to a multiple, run the network, return logits at the input H×W.

    Encoders with output stride 32 (U-Net R34) or 16 need sizes divisible by
    that stride. Padding is on the right and bottom with 0, which is the mean
    colour after ImageNet normalisation. If the network returns a smaller map
    (e.g. SegFormer at 1/4 size), it is upsampled bilinearly to the padded
    size before cropping, so the crop lines up with the original pixels.
    """

    def __init__(self, net: nn.Module, multiple: int) -> None:
        super().__init__()
        if multiple < 1:
            raise ValueError(f"multiple must be >= 1 (got {multiple})")
        self.net = net
        self.multiple = multiple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B, C, H, W] input (got shape {tuple(x.shape)})")
        h, w = x.shape[-2:]
        ph = -h % self.multiple
        pw = -w % self.multiple
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), value=0.0)
        out = self.net(x)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out[..., :h, :w]
