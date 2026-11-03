"""Leakage-free split assignment (T3).

Rules (rationale and evidence: docs/handoffs/2026-11-03-T3-lemasc.md):

1. Test = official LIP val + official CIHP val (D4), plus a DFMM test split.
2. Group key: LIP `<photo>_<person>` -> photo (crops of one photo stay
   together); CIHP -> image id; DFMM -> outfit id (all views of one outfit).
3. Groups joined by a near-duplicate pair are merged (union-find), within and
   across datasets.
4. A merged group that contains an official test image keeps its test images;
   all its other images are excluded (`leak_test`). The test set itself is
   never changed by this rule.
5. Other groups get a split from a seeded hash of the group key:
   LIP/CIHP train pool -> train / val; DFMM -> train / val / test.
6. Images whose mask has only background are excluded everywhere (`empty_mask`).
"""

from __future__ import annotations

import hashlib

import numpy as np

from cloth_segment.data.sources import dfmm_outfit_id

SPLIT_VERSION = "splits-v1"
SEED = 0
VAL_FRAC = {"lip": 0.05, "cihp": 0.05, "dfmm": 0.10}
TEST_FRAC = {"dfmm": 0.10}  # LIP/CIHP use their official val as test

# Near-duplicate rule: dHash-64 distance <= HASH_MAX_DIST (or vs the mirrored
# image), then 32×32 thumbnail correlation >= threshold. Any pair involving
# DFMM needs a much higher threshold: studio shots of the same model/pose in
# different clothes correlate up to ~0.98, and its plain backgrounds give
# false matches with LIP/CIHP up to ~0.73.
HASH_MAX_DIST = 6
CORR_MIN = 0.70
CORR_MIN_DFMM_PAIR = 0.98


def group_key(dataset: str, id_: str) -> str:
    if dataset == "lip":
        return "lip:" + id_.split("_")[0]
    if dataset == "cihp":
        return "cihp:" + id_
    if dataset == "dfmm":
        return "dfmm:" + dfmm_outfit_id(id_)
    raise ValueError(dataset)


def is_duplicate(ds_a: str, ds_b: str, corr: float,
                 corr_min: float = CORR_MIN, corr_min_dfmm: float = CORR_MIN_DFMM_PAIR) -> bool:
    limit = corr_min_dfmm if "dfmm" in (ds_a, ds_b) else corr_min
    return corr >= limit


def duplicate_mask(ds_a: np.ndarray, ds_b: np.ndarray, corr: np.ndarray,
                   corr_min: float = CORR_MIN, corr_min_dfmm: float = CORR_MIN_DFMM_PAIR) -> np.ndarray:
    """Vectorised is_duplicate over candidate pairs."""
    involves_dfmm = (ds_a == "dfmm") | (ds_b == "dfmm")
    return corr >= np.where(involves_dfmm, corr_min_dfmm, corr_min)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # deterministic root


def unit_hash(key: str, seed: int = SEED) -> float:
    """Deterministic uniform [0, 1) from a string; independent of row order."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign(datasets: list[str], source_splits: list[str], groups: list[str],
           empty: np.ndarray, seed: int = SEED) -> tuple[list[str], list[str]]:
    """Return (split, exclude_reason) per row. `groups` are merged group roots."""
    n = len(datasets)
    has_test = {g for g, ds, s in zip(groups, datasets, source_splits) if s == "val" and ds in ("lip", "cihp")}
    mixed = {}
    for g, ds in zip(groups, datasets):
        mixed.setdefault(g, set()).add(ds == "dfmm")
    split, reason = [""] * n, [""] * n
    for i, (ds, src, g) in enumerate(zip(datasets, source_splits, groups)):
        if ds in ("lip", "cihp") and src == "val":
            split[i] = "test"
        elif g in has_test:
            split[i], reason[i] = "excluded", "leak_test"
        else:
            if len(mixed[g]) > 1:
                raise ValueError(f"group {g} mixes DFMM and LIP/CIHP train images; no rule for that")
            u = unit_hash(g, seed)
            t = TEST_FRAC.get(ds, 0.0)
            split[i] = "test" if u < t else "val" if u < t + VAL_FRAC[ds] else "train"
        if empty[i] and split[i] != "excluded":
            split[i], reason[i] = "excluded", "empty_mask"
    return split, reason


def apply_rules(df, dup_a, dup_b, seed: int = SEED):
    """(group, split, exclude_reason) lists for manifest rows, given duplicate pairs.

    `df` needs dataset, source_split, id and c1..c11 (for empty masks);
    `dup_a`/`dup_b` are row positions of accepted duplicate pairs.
    """
    uf = UnionFind()
    base = [group_key(d, i) for d, i in zip(df.dataset, df.id)]
    for g in base:
        uf.find(g)
    for i, j in zip(dup_a, dup_b):
        uf.union(base[i], base[j])
    groups = [uf.find(g) for g in base]
    fg = [c for c in df.columns if c.startswith("c") and c[1:].isdigit() and c not in ("c0", "c255")]
    empty = (df[fg].sum(axis=1) == 0).to_numpy()
    split, reason = assign(list(df.dataset), list(df.source_split), groups, empty, seed)
    return groups, split, reason


def load_manifest(version: str = SPLIT_VERSION):
    """(manifest DataFrame, meta dict) for a built split version. Excluded rows included."""
    import json

    import pandas as pd

    from cloth_segment.paths import DATA

    root = DATA / "splits" / version
    if not (root / "DONE").exists():
        raise FileNotFoundError(f"{root} is not built; run scripts/build_splits.py")
    meta = json.loads((root / "meta.json").read_text())
    df = pd.read_csv(root / "manifest.csv", keep_default_na=False, dtype={"id": str, "exclude_reason": str})
    return df, meta
