"""Which raw classes merge into each shared class, and at what ratio (M5).

Pure table logic for the T1 EDA notebook. It reads the mapping from
`labels.py` and never changes it.
"""

from __future__ import annotations

import pandas as pd

from cloth_segment.data.labels import IGNORE, MAPPING, SHARED_CLASSES
from cloth_segment.data.sources import RAW_LABELS

COLUMNS = [
    "dataset",
    "shared",
    "raw_id",
    "raw",
    "pixels",
    "images",
    "pct_of_shared",
    "pct_of_dataset",
]


def merge_composition(counts: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """One row per raw class: where it goes and how big its part is.

    `counts` has one row per image and a column `v<raw id>` of pixel counts for
    every raw id of `dataset` (the `scripts/eda_scan.py` format).

    - `pct_of_shared`: share of the target class's pixels that this raw class
      supplies (100 = the only source). Sums to 100 per shared class.
    - `pct_of_dataset`: share of all pixels of the dataset. Sums to 100.
    Raw classes with no pixels stay in the table with 0.
    """
    rows = []
    for rid, name in RAW_LABELS[dataset].items():
        col = f"v{rid}"
        if col not in counts:
            raise KeyError(f"{dataset}: scan has no column {col}")
        target = MAPPING[dataset][rid]
        rows.append(
            {
                "dataset": dataset,
                "shared": "ignore" if target == IGNORE else SHARED_CLASSES[target],
                "raw_id": rid,
                "raw": name,
                "pixels": int(counts[col].sum()),
                "images": int((counts[col] > 0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    total = out["pixels"].sum()
    per_shared = out.groupby("shared")["pixels"].transform("sum")
    out["pct_of_shared"] = 100 * out["pixels"] / per_shared.where(per_shared > 0)
    out["pct_of_dataset"] = 100 * out["pixels"] / total if total else 0.0
    out["pct_of_shared"] = out["pct_of_shared"].fillna(0.0)
    order = {n: i for i, n in enumerate([*SHARED_CLASSES, "ignore"])}
    out = out.sort_values(
        ["shared", "pixels"], key=lambda s: s.map(order) if s.name == "shared" else -s
    )
    return out[COLUMNS].reset_index(drop=True)
