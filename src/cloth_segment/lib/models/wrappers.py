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

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the right and bottom of [B, C, H, W] with 0 up to the multiple."""
        if x.ndim != 4:
            raise ValueError(f"expected [B, C, H, W] input (got shape {tuple(x.shape)})")
        ph = -x.shape[-2] % self.multiple
        pw = -x.shape[-1] % self.multiple
        return F.pad(x, (0, pw, 0, ph), value=0.0) if ph or pw else x

    @staticmethod
    def restore(out: torch.Tensor, padded_hw, hw) -> torch.Tensor:
        """Upsample `out` to the padded size if smaller, then crop to the input `hw`."""
        if tuple(out.shape[-2:]) != tuple(padded_hw):
            out = F.interpolate(out, size=tuple(padded_hw), mode="bilinear", align_corners=False)
        return out[..., : hw[0], : hw[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        xp = self.pad(x)
        return self.restore(self.net(xp), xp.shape[-2:], hw)
