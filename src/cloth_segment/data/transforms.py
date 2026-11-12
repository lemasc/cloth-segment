"""Image/mask transforms for training and evaluation.

Same transforms for every model; parameters come from config, so they are
part of the run hash. Masks are only ever resized with nearest
neighbour, and every padded mask pixel is IGNORE.

- Train: scale the long side to `size * s` with `s ~ U(scale)`, horizontal
  flip, colour jitter, then take a random `size × size` window. A side shorter
  than `size` is placed at a random offset on a padded canvas.
- Eval: scale the long side to `size`, pad bottom/right to `size × size`. The
  sample keeps `valid_hw` (the unpadded part) and `orig_hw`, so predictions
  can be cropped and resized back to the original mask (evaluation runs at
  the original resolution).

Left/right classes are merged in the label set, so a flip needs no label
swap. Images are normalised with ImageNet statistics for every model, by us
rather than by a library's image processor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

from cloth_segment.data.labels import IGNORE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class AugmentConfig:
    """Training-time augmentation. Factors follow torchvision ColorJitter (0 = off)."""

    scale: tuple[float, float] = (1.0, 1.0)
    hflip: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0

    @classmethod
    def from_dict(cls, d: dict | None) -> AugmentConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown augment keys: {sorted(unknown)}")
        if "scale" in d:
            lo, hi = d["scale"]
            if not 0 < lo <= hi:
                raise ValueError(f"augment.scale must satisfy 0 < lo <= hi, got {d['scale']}")
            d["scale"] = (float(lo), float(hi))
        if not 0 <= d.get("hflip", 0.0) <= 1:
            raise ValueError("augment.hflip is a probability")
        if not 0 <= d.get("hue", 0.0) <= 0.5:
            raise ValueError("augment.hue must be in [0, 0.5]")
        return cls(**d)


def _uniform(lo: float, hi: float, g: torch.Generator | None) -> float:
    return lo + (hi - lo) * torch.rand((), generator=g).item()


def _randint(n: int, g: torch.Generator | None) -> int:
    """Uniform integer in [0, n]."""
    return int(torch.randint(0, n + 1, (), generator=g).item())


def _resize(image: torch.Tensor, mask: torch.Tensor, h: int, w: int):
    image = F.resize(image, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)
    mask = F.resize(mask[None], [h, w], interpolation=InterpolationMode.NEAREST_EXACT)[0]
    return image, mask


def _scaled_hw(h: int, w: int, long_side: float) -> tuple[int, int]:
    r = long_side / max(h, w)
    return max(1, round(h * r)), max(1, round(w * r))


def normalize(image: torch.Tensor) -> torch.Tensor:
    """uint8 (3, H, W) -> float32, ImageNet-normalised."""
    x = image.float().div_(255)
    return F.normalize(x, IMAGENET_MEAN, IMAGENET_STD)


def color_jitter(image: torch.Tensor, aug: AugmentConfig, g: torch.Generator | None):
    """Brightness, contrast, saturation, hue in a fixed order on a uint8 image."""
    if aug.brightness:
        image = F.adjust_brightness(image, _uniform(1 - aug.brightness, 1 + aug.brightness, g))
    if aug.contrast:
        image = F.adjust_contrast(image, _uniform(1 - aug.contrast, 1 + aug.contrast, g))
    if aug.saturation:
        image = F.adjust_saturation(image, _uniform(1 - aug.saturation, 1 + aug.saturation, g))
    if aug.hue:
        image = F.adjust_hue(image, _uniform(-aug.hue, aug.hue, g))
    return image


def _window(length: int, size: int, g: torch.Generator | None) -> tuple[int, int]:
    """(src_start, dst_start) of a `size` window on an axis of `length` pixels."""
    if length >= size:
        return _randint(length - size, g), 0
    return 0, _randint(size - length, g)


def train_transform(
    image: torch.Tensor,
    mask: torch.Tensor,
    size: int,
    aug: AugmentConfig,
    g: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 image (3, H, W) and mask (H, W) -> float (3, S, S), int64 (S, S)."""
    h, w = mask.shape
    nh, nw = _scaled_hw(h, w, size * _uniform(*aug.scale, g))
    image, mask = _resize(image, mask, nh, nw)
    if aug.hflip and torch.rand((), generator=g).item() < aug.hflip:
        image, mask = image.flip(-1), mask.flip(-1)

    (sy, dy), (sx, dx) = _window(nh, size, g), _window(nw, size, g)
    ch, cw = min(nh, size), min(nw, size)
    image = image[:, sy : sy + ch, sx : sx + cw]
    mask = mask[sy : sy + ch, sx : sx + cw]
    image = normalize(color_jitter(image, aug, g))

    out_img = torch.zeros(3, size, size, dtype=torch.float32)
    out_mask = torch.full((size, size), IGNORE, dtype=torch.int64)
    out_img[:, dy : dy + ch, dx : dx + cw] = image
    out_mask[dy : dy + ch, dx : dx + cw] = mask
    return out_img, out_mask


def eval_transform(
    image: torch.Tensor, mask: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Deterministic: long side -> `size`, pad bottom/right. Returns (image, mask, valid_hw)."""
    h, w = mask.shape
    nh, nw = _scaled_hw(h, w, size)
    image, mask = _resize(image, mask, nh, nw)
    out_img = torch.zeros(3, size, size, dtype=torch.float32)
    out_mask = torch.full((size, size), IGNORE, dtype=torch.int64)
    out_img[:, :nh, :nw] = normalize(image)
    out_mask[:nh, :nw] = mask
    return out_img, out_mask, (nh, nw)
