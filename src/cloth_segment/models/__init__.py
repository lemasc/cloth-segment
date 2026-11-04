"""Model builders: config (name + params) -> nn.Module.

Tasks T6 (U-Net R34), T7 (DeepLabV3+ R50), T8 (SegFormer-B2); E1 adds attribute heads.

Every model takes [B, 3, H, W] (ImageNet-normalised by the data transforms) and
returns raw logits [B, NUM_CLASSES, H, W] at the input size, without softmax.

To add a model: write a builder in its own module, decorate it with
`@register("<name>")`, and import that module at the bottom of this file.
The keys under `model:` in the config, except `name`, are passed to the
builder as keyword arguments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import nn

Builder = Callable[..., nn.Module]

MODELS: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    """Decorator that adds a builder to MODELS under `name`."""

    def deco(fn: Builder) -> Builder:
        if name in MODELS:
            raise ValueError(f"model {name!r} is already registered")
        MODELS[name] = fn
        return fn

    return deco


def build_model(cfg: Mapping[str, Any]) -> nn.Module:
    """Build the model named by cfg["model"]["name"] with the other model.* keys as kwargs."""
    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("name", None)
    if name not in MODELS:
        raise ValueError(f"unknown model.name {name!r}; known: {sorted(MODELS)}")
    return MODELS[name](**model_cfg)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters (all, or only those with requires_grad)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# Model modules register themselves on import (T6: unet, T7: deeplab, T8: segformer).
from cloth_segment.models import unet  # noqa: E402, F401  (import registers the builder)
