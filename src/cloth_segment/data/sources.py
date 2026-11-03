"""Raw dataset sources: where image/mask pairs live and what raw label IDs mean.

This module only describes the datasets as downloaded. Mapping to the shared
label set is T2 (`labels.py`), splitting is T3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cloth_segment.paths import DATASETS

_LIP_CIHP_COMMON = {
    0: "background", 1: "hat", 2: "hair", 3: "glove", 4: "sunglasses",
    5: "upper_clothes", 6: "dress", 7: "coat", 8: "socks", 9: "pants",
    11: "scarf", 12: "skirt", 13: "face", 14: "left_arm", 15: "right_arm",
    16: "left_leg", 17: "right_leg", 18: "left_shoe", 19: "right_shoe",
}

RAW_LABELS: dict[str, dict[int, str]] = {
    "lip": {**_LIP_CIHP_COMMON, 10: "jumpsuits"},
    "cihp": {**_LIP_CIHP_COMMON, 10: "torso_skin"},
    "dfmm": {
        0: "background", 1: "top", 2: "outer", 3: "skirt", 4: "dress",
        5: "pants", 6: "leggings", 7: "headwear", 8: "eyeglass",
        9: "neckwear", 10: "belt", 11: "footwear", 12: "bag", 13: "hair",
        14: "face", 15: "skin", 16: "ring", 17: "wrist_wearing", 18: "socks",
        19: "gloves", 20: "necklace", 21: "rompers", 22: "earrings", 23: "tie",
    },
}
RAW_LABELS = {k: dict(sorted(v.items())) for k, v in RAW_LABELS.items()}

# Splits with labels. LIP/CIHP official test sets have no masks; DFMM has no
# official split (one pool, split by outfit in T3).
SPLITS: dict[str, tuple[str, ...]] = {
    "lip": ("train", "val"),
    "cihp": ("train", "val"),
    "dfmm": ("all",),
}


@dataclass(frozen=True)
class Pair:
    dataset: str
    split: str
    id: str
    image: Path
    mask: Path


def _lip(split: str, root: Path) -> list[Pair]:
    ids = (root / f"{split}_id.txt").read_text().split()
    img_dir = root / "TrainVal_images" / f"{split}_images"
    mask_dir = root / "TrainVal_parsing_annotations" / f"{split}_segmentations"
    return [Pair("lip", split, i, img_dir / f"{i}.jpg", mask_dir / f"{i}.png") for i in ids]


def _cihp(split: str, root: Path) -> list[Pair]:
    folder = {"train": "Training", "val": "Validation"}[split]
    base = root / folder
    ids = (base / f"{split}_id.txt").read_text().split()
    return [
        Pair("cihp", split, i, base / "Images" / f"{i}.jpg", base / "Category_ids" / f"{i}.png")
        for i in ids
    ]


def _dfmm(split: str, root: Path) -> list[Pair]:
    pairs = []
    for m in sorted((root / "segm").glob("*_segm.png")):
        stem = m.name.removesuffix("_segm.png")
        pairs.append(Pair("dfmm", split, stem, root / "images" / f"{stem}.jpg", m))
    return pairs


_ROOTS = {"lip": "lip", "cihp": "cihp", "dfmm": "deepfashion-multimodal"}
_LISTERS = {"lip": _lip, "cihp": _cihp, "dfmm": _dfmm}


def dataset_root(dataset: str) -> Path:
    return DATASETS / _ROOTS[dataset]


def list_pairs(dataset: str, split: str) -> list[Pair]:
    """All labelled image/mask pairs of one dataset split, in a stable order."""
    if split not in SPLITS[dataset]:
        raise ValueError(f"{dataset} has no labelled split {split!r}; choose from {SPLITS[dataset]}")
    return _LISTERS[dataset](split, dataset_root(dataset))


def dfmm_outfit_id(stem: str) -> str:
    """`MEN-Denim-id_00000080-01_7_additional` -> `id_00000080`.

    The same outfit appears in several views (`-01`, `-02`, ... and pose
    suffixes), so this is the group key for leakage-free splits.
    """
    for part in stem.split("-"):
        if part.startswith("id_"):
            return part
    raise ValueError(f"no outfit id in {stem!r}")
