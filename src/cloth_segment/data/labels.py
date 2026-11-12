"""Shared label set and per-dataset raw -> shared mapping.

Every raw ID of every dataset is mapped explicitly, either to a shared class or
to IGNORE. Conversion fails on any value that is not in the raw label set, so
a wrong source folder or a corrupt mask cannot silently become background.

Left/right parts are merged, so a horizontal flip needs no label swap.

Changing anything here changes the processed data: bump LABEL_VERSION.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from cloth_segment.data.sources import RAW_LABELS

LABEL_VERSION = "labels-v1"

SHARED_CLASSES: tuple[str, ...] = (
    "background",
    "hair",
    "face",
    "skin",
    "upper",
    "outer",
    "dress",
    "skirt",
    "pants",
    "footwear",
    "headwear",
    "accessory",
)
NUM_CLASSES = len(SHARED_CLASSES)
IGNORE = 255
IGNORE_NAME = "ignore"

# Display colours (RGB) for overlays and figures; index = shared class ID.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),  # background
    (128, 64, 0),  # hair
    (255, 200, 150),  # face
    (230, 140, 100),  # skin
    (220, 30, 30),  # upper
    (30, 90, 220),  # outer
    (200, 40, 200),  # dress
    (255, 160, 0),  # skirt
    (30, 160, 60),  # pants
    (90, 90, 90),  # footwear
    (0, 200, 200),  # headwear
    (240, 240, 0),  # accessory
)
IGNORE_COLOR = (255, 255, 255)

_LIP_CIHP_COMMON = {
    "background": "background",
    "hat": "headwear",
    "hair": "hair",
    "glove": "accessory",
    "sunglasses": "accessory",
    "upper_clothes": "upper",
    "dress": "dress",
    "coat": "outer",
    "socks": "footwear",
    "pants": "pants",
    "scarf": "accessory",
    "skirt": "skirt",
    "face": "face",
    "left_arm": "skin",
    "right_arm": "skin",
    "left_leg": "skin",
    "right_leg": "skin",
    "left_shoe": "footwear",
    "right_shoe": "footwear",
}

# raw name -> shared name (or IGNORE_NAME), per dataset
MAPPING_BY_NAME: dict[str, dict[str, str]] = {
    "lip": {**_LIP_CIHP_COMMON, "jumpsuits": "dress"},
    "cihp": {**_LIP_CIHP_COMMON, "torso_skin": "skin"},
    "dfmm": {
        "background": "background",
        "top": "upper",
        "outer": "outer",
        "skirt": "skirt",
        "dress": "dress",
        "pants": "pants",
        "leggings": "footwear",  # knee/thigh-high socks and tights, like LIP/CIHP socks
        "headwear": "headwear",
        "eyeglass": "accessory",
        "neckwear": "accessory",
        "belt": IGNORE_NAME,  # no belt class in LIP/CIHP (labelled as garment there)
        "footwear": "footwear",
        "bag": IGNORE_NAME,  # no bag class in LIP/CIHP (labelled as background there)
        "hair": "hair",
        "face": "face",
        "skin": "skin",
        "ring": IGNORE_NAME,  # jewellery: tiny, no LIP/CIHP counterpart
        "wrist_wearing": IGNORE_NAME,
        "socks": "footwear",
        "gloves": "accessory",
        "necklace": IGNORE_NAME,
        "rompers": "dress",
        "earrings": IGNORE_NAME,
        "tie": "accessory",
    },
}


def _shared_id(name: str) -> int:
    return IGNORE if name == IGNORE_NAME else SHARED_CLASSES.index(name)


def _build_mapping() -> dict[str, dict[int, int]]:
    out = {}
    for ds, raw in RAW_LABELS.items():
        by_name = MAPPING_BY_NAME[ds]
        missing = set(raw.values()) - set(by_name)
        extra = set(by_name) - set(raw.values())
        if missing or extra:
            raise RuntimeError(f"{ds}: unmapped raw labels {missing}, unknown names {extra}")
        out[ds] = {rid: _shared_id(by_name[name]) for rid, name in raw.items()}
    return out


# raw ID -> shared ID, per dataset
MAPPING: dict[str, dict[int, int]] = _build_mapping()

_UNMAPPED = -1


def _lut(dataset: str) -> np.ndarray:
    lut = np.full(256, _UNMAPPED, dtype=np.int16)
    for rid, sid in MAPPING[dataset].items():
        lut[rid] = sid
    return lut


_LUTS = {ds: _lut(ds) for ds in MAPPING}


class UnknownLabelError(ValueError):
    pass


def convert(mask: np.ndarray, dataset: str) -> np.ndarray:
    """Map a raw label mask (H×W integer array) to shared IDs (uint8, IGNORE=255).

    Raises UnknownLabelError if the mask contains a value outside the
    dataset's raw label set.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D (got shape {mask.shape}); load masks as L/P, never RGB")
    if mask.dtype != np.uint8:
        if mask.min() < 0 or mask.max() > 255:
            raise UnknownLabelError(f"{dataset}: mask values outside 0..255")
        mask = mask.astype(np.uint8)
    out = _LUTS[dataset][mask]
    if (out == _UNMAPPED).any():
        bad = sorted(set(np.unique(mask[out == _UNMAPPED]).tolist()))
        raise UnknownLabelError(f"{dataset}: unknown raw label values {bad}")
    return out.astype(np.uint8)


def colorize(shared_mask: np.ndarray) -> np.ndarray:
    """Shared-ID mask -> H×W×3 uint8 RGB for display."""
    table = np.zeros((256, 3), dtype=np.uint8)
    table[:NUM_CLASSES] = PALETTE
    table[IGNORE] = IGNORE_COLOR
    return table[shared_mask]


def label_spec() -> dict:
    """Everything that defines the shared labels, for writing next to processed data."""
    return {
        "version": LABEL_VERSION,
        "classes": list(SHARED_CLASSES),
        "ignore": IGNORE,
        "mapping": {ds: {str(k): v for k, v in m.items()} for ds, m in MAPPING.items()},
    }


def label_hash() -> str:
    """Short hash of label_spec(); changes whenever the mapping changes."""
    blob = json.dumps(label_spec(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
