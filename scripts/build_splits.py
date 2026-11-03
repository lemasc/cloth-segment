"""Build the leakage-free splits and the manifest (T3). Needs prepare_masks.py first.

Writes, under data/splits/<SPLIT_VERSION>/:

    manifest.csv      one row per labelled pair (see below)
    duplicates.csv    accepted near-duplicate pairs (evidence for the report)
    meta.json         rules, counts, label hash, data_version
    DONE

manifest.csv columns: dataset, source_split, id, split (train/val/test/
excluded), exclude_reason, group, image (relative to datasets/), mask
(relative to data/), h, w, c0..c11, c255 (pixels per shared class), and the
18 raw DFMM attribute codes (-1 for non-DFMM rows or missing labels).

data_version = <SPLIT_VERSION>-<sha256(manifest.csv)[:12]>; runs record it (D7).
Frozen like the masks: same content -> no-op, different content -> refuse.

Usage: uv run python scripts/build_splits.py [--workers 16]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from cloth_segment.data import splits as S
from cloth_segment.data.dedup import hash_file, near_pairs, thumb_corr, thumbnail, to_u64
from cloth_segment.data.labels import IGNORE, LABEL_VERSION, NUM_CLASSES, label_hash
from cloth_segment.data.sources import (
    DFMM_ATTRS, MISSING_ATTR, SPLITS, list_pairs, load_dfmm_attributes,
)
from cloth_segment.paths import DATA, DATASETS

MASKS = DATA / "masks" / LABEL_VERSION
OUT = DATA / "splits" / S.SPLIT_VERSION
COUNT_COLS = [f"c{i}" for i in range(NUM_CLASSES)] + [f"c{IGNORE}"]


def _hash_and_thumb(path):
    h, hf = hash_file(path)
    return h, hf, thumbnail(path)


def find_duplicates(df: pd.DataFrame, paths: list, workers: int) -> pd.DataFrame:
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_hash_and_thumb, paths, chunksize=128))
    h = to_u64(np.stack([r[0] for r in res]))
    hf = to_u64(np.stack([r[1] for r in res]))
    thumbs = np.stack([r[2] for r in res])
    print(f"  hashed {len(h)} images", flush=True)

    plain = near_pairs(h, max_dist=S.HASH_MAX_DIST)
    flip = near_pairs(h, hf, max_dist=S.HASH_MAX_DIST)
    flip = flip[flip[:, 0] != flip[:, 1]]
    cand = pd.concat([
        pd.DataFrame({"a": plain[:, 0], "b": plain[:, 1], "hash_dist": plain[:, 2], "flip": False}),
        pd.DataFrame({"a": flip[:, :2].min(1), "b": flip[:, :2].max(1), "hash_dist": flip[:, 2], "flip": True}),
    ], ignore_index=True)
    a, b = cand.a.to_numpy(), cand.b.to_numpy()
    tb = thumbs[b]
    tb = np.where(cand.flip.to_numpy()[:, None, None], tb[:, :, ::-1], tb)
    cand["corr"] = thumb_corr(thumbs[a], tb)
    cand = cand.sort_values("corr", ascending=False).drop_duplicates(["a", "b"])
    ds = df.dataset.to_numpy()
    keep = [S.is_duplicate(ds[i], ds[j], c) for i, j, c in zip(cand.a, cand.b, cand["corr"])]
    dup = cand[keep].reset_index(drop=True)
    print(f"  {len(cand)} hash candidates -> {len(dup)} duplicates", flush=True)
    for side in ("a", "b"):
        for col in ("dataset", "source_split", "id"):
            dup[f"{col}_{side}"] = df[col].to_numpy()[dup[side]]
    return dup


def build(workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (MASKS / "DONE").exists():
        raise SystemExit(f"{MASKS} missing; run scripts/prepare_masks.py first")
    if json.loads((MASKS / "labels.json").read_text())["hash"] != label_hash():
        raise SystemExit(f"{MASKS} was built with a different label mapping; bump LABEL_VERSION")

    pairs = [p for ds, ss in SPLITS.items() for s in ss for p in list_pairs(ds, s)]
    df = pd.DataFrame({
        "dataset": [p.dataset for p in pairs],
        "source_split": [p.split for p in pairs],
        "id": [p.id for p in pairs],
        "image": [str(p.image.relative_to(DATASETS)) for p in pairs],
        "mask": [f"masks/{LABEL_VERSION}/{p.dataset}/{p.split}/{p.id}.png" for p in pairs],
    })
    stats = pd.read_csv(MASKS / "stats.csv", keep_default_na=False, dtype={"id": str})
    df = df.merge(stats, left_on=["dataset", "source_split", "id"], right_on=["dataset", "split", "id"],
                  how="left", validate="one_to_one").drop(columns="split")
    if df.h.isna().any():
        raise SystemExit("some pairs have no entry in masks stats.csv")

    dup = find_duplicates(df, [p.image for p in pairs], workers)
    uf = S.UnionFind()
    base = [S.group_key(d, i) for d, i in zip(df.dataset, df.id)]
    for g in base:
        uf.find(g)
    for i, j in zip(dup.a, dup.b):
        uf.union(base[i], base[j])
    df["group"] = [uf.find(g) for g in base]

    empty = (df[[f"c{i}" for i in range(1, NUM_CLASSES)]].sum(axis=1) == 0).to_numpy()
    df["split"], df["exclude_reason"] = S.assign(list(df.dataset), list(df.source_split), list(df.group), empty)

    attrs = load_dfmm_attributes()
    df = df.join(attrs, on="id")
    df.loc[df.dataset != "dfmm", list(DFMM_ATTRS)] = MISSING_ATTR
    df[list(DFMM_ATTRS)] = df[list(DFMM_ATTRS)].fillna(MISSING_ATTR).astype(int)

    cols = ["dataset", "source_split", "id", "split", "exclude_reason", "group", "image", "mask",
            "h", "w", *COUNT_COLS, *DFMM_ATTRS]
    return df[cols], dup.drop(columns=["a", "b"])


def check(df: pd.DataFrame) -> None:
    """Leakage invariants; raise if any is violated."""
    kept = df[df.split != "excluded"]
    per_group = kept.groupby("group").split.nunique()
    assert (per_group == 1).all(), f"groups in >1 split: {list(per_group[per_group > 1].index[:5])}"
    official = df[(df.dataset != "dfmm") & (df.source_split == "val")]
    assert (official.split.isin(["test", "excluded"])).all()
    assert not ((df.dataset != "dfmm") & (df.source_split == "train") & (df.split == "test")).any()
    assert (df[df.split == "excluded"].exclude_reason != "").all()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    a = ap.parse_args()

    df, dup = build(a.workers)
    check(df)
    csv_bytes = df.to_csv(index=False).encode()
    sha = hashlib.sha256(csv_bytes).hexdigest()
    data_version = f"{S.SPLIT_VERSION}-{sha[:12]}"

    if OUT.exists():
        old = json.loads((OUT / "meta.json").read_text()).get("data_version") if (OUT / "DONE").exists() else None
        if old == data_version:
            print(f"{OUT} is up to date ({data_version}); nothing to do.")
            return
        raise SystemExit(f"{OUT} exists as {old}, new build is {data_version}: bump SPLIT_VERSION instead of overwriting.")

    counts = df.groupby(["dataset", "split"]).size().unstack(fill_value=0)
    reasons = df[df.split == "excluded"].groupby(["dataset", "exclude_reason"]).size()
    meta = {
        "data_version": data_version,
        "manifest_sha256": sha,
        "label_version": LABEL_VERSION,
        "label_hash": label_hash(),
        "seed": S.SEED,
        "val_frac": S.VAL_FRAC,
        "test_frac": S.TEST_FRAC,
        "duplicate_rule": {"hash": "dhash64 (+mirrored)", "hash_max_dist": S.HASH_MAX_DIST,
                           "corr_min": S.CORR_MIN, "corr_min_dfmm_pair": S.CORR_MIN_DFMM_PAIR},
        "counts": {ds: row.to_dict() for ds, row in counts.iterrows()},
        "excluded": {f"{d}/{r}": int(n) for (d, r), n in reasons.items()},
        "duplicate_pairs": len(dup),
    }
    tmp = OUT.with_name(f".{OUT.name}.tmp-{os.getpid()}")
    tmp.mkdir(parents=True)
    try:
        (tmp / "manifest.csv").write_bytes(csv_bytes)
        dup.to_csv(tmp / "duplicates.csv", index=False)
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
        (tmp / "DONE").touch()
        tmp.rename(OUT)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(counts.to_string())
    print(json.dumps(meta["excluded"], indent=1))
    print(f"-> {OUT} ({data_version})")


if __name__ == "__main__":
    main()
