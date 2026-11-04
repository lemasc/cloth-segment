"""Model builders: config (name + params) -> nn.Module.

Tasks T6 (U-Net R34), T7 (DeepLabV3+ R50), T8 (SegFormer-B2); E1 adds attribute heads.

Contract used by `train.py` and `evaluate.py` (T5): a built model maps a
normalised image batch (B, 3, S, S) to a plain logits tensor (B, NUM_CLASSES,
h, w). `h, w` may be smaller than `S` (SegFormer predicts at 1/4); callers
upsample with `logits_at`. A builder takes the `model` config section and
registers itself with `@register("<name>")`.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

Builder = Callable[[dict], nn.Module]
REGISTRY: dict[str, Builder] = {}

# Names the plan expects; each is registered by its own task.
PLANNED = {"deeplabv3p_r50": "T7", "segformer_b2": "T8"}


def register(name: str) -> Callable[[Builder], Builder]:
    def deco(fn: Builder) -> Builder:
        if name in REGISTRY:
            raise ValueError(f"model {name!r} registered twice")
        REGISTRY[name] = fn
        return fn

    return deco


def build_model(model_cfg: dict) -> nn.Module:
    """Build the model named by `model_cfg["name"]`."""
    name = (model_cfg or {}).get("name")
    if not name:
        raise ValueError("config has no model.name")
    if name not in REGISTRY:
        if name in PLANNED:
            raise NotImplementedError(f"model {name!r} is not built yet ({PLANNED[name]})")
        raise ValueError(f"unknown model {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](model_cfg)


def logits_at(logits: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Bilinearly upsample (B, C, h, w) logits to `hw` if they are smaller."""
    if tuple(logits.shape[-2:]) == tuple(hw):
        return logits
    return F.interpolate(logits, size=tuple(hw), mode="bilinear", align_corners=False)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters (all, or only those with requires_grad)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# Model modules register themselves on import (T6: unet; T7, T8 add theirs here).
from cloth_segment.models import unet  # noqa: E402, F401  (import registers the builder)
