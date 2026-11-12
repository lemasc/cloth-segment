"""PyTorch Dataset over the split manifest: images, masks and DFMM attributes.

Reads original images from `datasets/` and labels-v1 masks from `data/`, via
the paths in the manifest. Nothing is pre-resized. Every sample carries the
DFMM attribute targets: NA and missing codes, and all non-DFMM images, get
IGNORE, so a multi-task head can mask its loss with `ignore_index`. Which
attributes become heads is decided by the config, not here.

Config keys used (see configs/base.yaml): `data.version`, `data.datasets`,
`data.sampling`, `input.size`, `augment.*`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from cloth_segment.lib import paths
from cloth_segment.lib.data.labels import IGNORE, NUM_CLASSES
from cloth_segment.lib.data.sources import DFMM_ATTRS, MISSING_ATTR
from cloth_segment.lib.data.transforms import AugmentConfig, eval_transform, train_transform

DATASET_NAMES: tuple[str, ...] = ("lip", "cihp", "dfmm")
ATTR_NAMES: tuple[str, ...] = tuple(DFMM_ATTRS)
ATTR_NUM_CLASSES: dict[str, int] = {k: len(v) - 1 for k, v in DFMM_ATTRS.items()}  # NA dropped
SPLITS = ("train", "val", "test")
SAMPLING = ("natural", "balanced")


def attribute_targets(codes: np.ndarray) -> np.ndarray:
    """Raw DFMM codes (..., len(ATTR_NAMES)) -> int64 targets; NA and MISSING_ATTR -> IGNORE."""
    codes = np.asarray(codes, dtype=np.int64)
    if codes.shape[-1] != len(ATTR_NAMES):
        raise ValueError(f"expected {len(ATTR_NAMES)} attribute columns, got {codes.shape[-1]}")
    na = np.array([ATTR_NUM_CLASSES[k] for k in ATTR_NAMES])
    bad = (codes < MISSING_ATTR) | (codes > na)
    if bad.any():
        raise ValueError("attribute code outside its value range")
    return np.where((codes == MISSING_ATTR) | (codes == na), IGNORE, codes)


def select_rows(df, split: str, datasets: Sequence[str]):
    """Manifest rows of one split for the given datasets, in manifest order."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    unknown = set(datasets) - set(DATASET_NAMES)
    if unknown:
        raise ValueError(f"unknown datasets: {sorted(unknown)}")
    rows = df[(df.split == split) & df.dataset.isin(list(datasets))]
    if (rows.exclude_reason != "").any():
        raise ValueError("manifest has excluded rows inside a split")
    return rows.reset_index(drop=True)


def read_image(path: Path) -> torch.Tensor:
    """uint8 (3, H, W); grayscale images are converted to RGB."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()


def read_mask(path: Path) -> torch.Tensor:
    """uint8 (H, W) shared-label mask; fails on values outside 0..NUM_CLASSES-1 and IGNORE."""
    with Image.open(path) as im:
        if im.mode != "L":
            raise ValueError(f"{path}: mask mode {im.mode}, expected L")
        arr = np.asarray(im)
    bad = (arr >= NUM_CLASSES) & (arr != IGNORE)
    if bad.any():
        raise ValueError(f"{path}: unexpected mask values {sorted(np.unique(arr[bad]))}")
    return torch.from_numpy(arr.copy())


class SegDataset(Dataset):
    """One manifest split. `train=True` applies augmentation, else the eval transform.

    Each sample is a dict: `image` float (3, S, S), `mask` int64 (S, S),
    `attrs` int64 (len(ATTR_NAMES),), `dataset` int (index in DATASET_NAMES),
    `index` int (row in `self.rows`), `orig_hw` and, for eval, `valid_hw`.
    """

    def __init__(
        self,
        rows,
        size: int,
        train: bool,
        augment: AugmentConfig | None = None,
        image_root: Path = paths.DATASETS,
        mask_root: Path = paths.DATA,
    ) -> None:
        if size <= 0 or size % 32:
            raise ValueError(f"input size must be a positive multiple of 32, got {size}")
        self.rows = rows.reset_index(drop=True)
        self.size = size
        self.train = train
        self.augment = augment or AugmentConfig()
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self._images = self.rows["image"].tolist()
        self._masks = self.rows["mask"].tolist()
        self._dataset = [DATASET_NAMES.index(d) for d in self.rows.dataset]
        self._attrs = torch.from_numpy(attribute_targets(self.rows[list(ATTR_NAMES)].to_numpy()))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        image = read_image(self.image_root / self._images[i])
        mask = read_mask(self.mask_root / self._masks[i])
        if image.shape[1:] != mask.shape:
            raise ValueError(
                f"{self._images[i]}: image {tuple(image.shape[1:])} != mask {tuple(mask.shape)}"
            )
        sample = {
            "attrs": self._attrs[i],
            "dataset": self._dataset[i],
            "index": i,
            "orig_hw": tuple(mask.shape),
        }
        if self.train:
            sample["image"], sample["mask"] = train_transform(image, mask, self.size, self.augment)
        else:
            sample["image"], sample["mask"], sample["valid_hw"] = eval_transform(
                image, mask, self.size
            )
        return sample


def collate(batch: list[dict]) -> dict:
    """Stack tensors and ints; keep (h, w) tuples as lists."""
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        elif isinstance(vals[0], int):
            out[key] = torch.tensor(vals)
        else:
            out[key] = vals
    return out


def sample_weights(datasets: Sequence[str], mode: str) -> np.ndarray:
    """Per-sample weights. `natural`: uniform; `balanced`: each dataset gets equal total weight."""
    if mode not in SAMPLING:
        raise ValueError(f"sampling must be one of {SAMPLING}, got {mode!r}")
    ds = np.asarray(datasets)
    if mode == "natural":
        return np.full(len(ds), 1.0 / len(ds))
    names, counts = np.unique(ds, return_counts=True)
    per = dict(zip(names, 1.0 / (counts * len(names))))
    return np.array([per[d] for d in ds])


def make_sampler(dataset: SegDataset, mode: str, generator: torch.Generator | None = None):
    """None for `natural` (use `shuffle=True`), else a WeightedRandomSampler of one epoch."""
    if mode == "natural":
        return None
    w = sample_weights(dataset.rows.dataset.tolist(), mode)
    return WeightedRandomSampler(
        torch.from_numpy(w), num_samples=len(w), replacement=True, generator=generator
    )


def class_pixel_counts(rows) -> np.ndarray:
    """Pixel count per shared class (c0..c11) over the given rows; use train rows only."""
    return rows[[f"c{k}" for k in range(NUM_CLASSES)]].to_numpy().sum(axis=0)


def load_split(cfg: dict, split: str, train: bool | None = None) -> SegDataset:
    """Build the dataset of one split from a resolved config.

    `train` defaults to `split == "train"`. Fails if the built manifest does
    not match `data.version`, so a run never silently uses other data.
    """
    from cloth_segment.lib.data.splits import load_manifest

    data = cfg["data"]
    version = data["version"]
    df, meta = load_manifest(version.rsplit("-", 1)[0])
    if meta["data_version"] != version:
        raise ValueError(f"config wants {version}, built manifest is {meta['data_version']}")
    train = split == "train" if train is None else train
    return SegDataset(
        select_rows(df, split, data["datasets"]),
        size=cfg["input"]["size"],
        train=train,
        augment=AugmentConfig.from_dict(cfg.get("augment")),
    )
