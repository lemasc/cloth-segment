"""Convert every raw mask to the shared label set (T2).

Writes, under data/masks/<LABEL_VERSION>/:

    labels.json                      label_spec() + label_hash
    <dataset>/<split>/<id>.png       shared-ID mask, mode L, same size as the image
    stats.csv                        one row per pair: dataset, split, id, h, w, c0..c11, c255
    DONE

The output is frozen (D8). It is built in a temporary folder and renamed when
complete. If the folder already exists with the same label hash this is a
no-op; with a different hash it fails: bump LABEL_VERSION in labels.py.

Usage: uv run python scripts/prepare_masks.py [--workers 16]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from cloth_segment.data.labels import (
    IGNORE,
    LABEL_VERSION,
    NUM_CLASSES,
    convert,
    label_hash,
    label_spec,
)
from cloth_segment.data.sources import SPLITS, Pair, list_pairs
from cloth_segment.paths import DATA

OUT = DATA / "masks" / LABEL_VERSION
COUNT_COLS = [f"c{i}" for i in range(NUM_CLASSES)] + [f"c{IGNORE}"]


def convert_pair(args: tuple[Pair, Path]) -> dict:
    pair, root = args
    with Image.open(pair.mask) as m:
        if m.mode not in ("L", "P"):
            raise ValueError(f"{pair.mask}: mask mode {m.mode}, expected L or P")
        raw = np.asarray(m)
    with Image.open(pair.image) as im:
        if im.size != (raw.shape[1], raw.shape[0]):
            raise ValueError(f"{pair.id}: image {im.size} vs mask {raw.shape[::-1]}")
    shared = convert(raw, pair.dataset)
    dst = root / pair.dataset / pair.split / f"{pair.id}.png"
    Image.fromarray(shared, mode="L").save(dst)
    counts = np.bincount(shared.ravel(), minlength=256)
    row = {
        "dataset": pair.dataset,
        "split": pair.split,
        "id": pair.id,
        "h": shared.shape[0],
        "w": shared.shape[1],
    }
    row.update({f"c{i}": int(counts[i]) for i in range(NUM_CLASSES)})
    row[f"c{IGNORE}"] = int(counts[IGNORE])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    a = ap.parse_args()

    spec = {**label_spec(), "hash": label_hash()}
    if OUT.exists():
        done = (OUT / "DONE").exists()
        old = json.loads((OUT / "labels.json").read_text()).get("hash") if done else None
        if old == spec["hash"]:
            print(f"{OUT} is up to date (hash {old}); nothing to do.")
            return
        raise SystemExit(
            f"{OUT} exists with hash {old} (complete={done}), current is {spec['hash']}. "
            "The label mapping changed: bump LABEL_VERSION instead of overwriting."
        )

    tmp = OUT.with_name(f".{OUT.name}.tmp-{os.getpid()}")
    pairs = [p for ds, splits in SPLITS.items() for s in splits for p in list_pairs(ds, s)]
    for ds, splits in SPLITS.items():
        for s in splits:
            (tmp / ds / s).mkdir(parents=True)
    try:
        with ProcessPoolExecutor(a.workers) as ex, (tmp / "stats.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, ["dataset", "split", "id", "h", "w", *COUNT_COLS])
            w.writeheader()
            for i, row in enumerate(ex.map(convert_pair, ((p, tmp) for p in pairs), chunksize=64)):
                w.writerow(row)
                if (i + 1) % 10000 == 0:
                    print(f"  {i + 1}/{len(pairs)}", flush=True)
        (tmp / "labels.json").write_text(json.dumps(spec, indent=2))
        (tmp / "DONE").touch()
        tmp.rename(OUT)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"{len(pairs)} masks -> {OUT} (hash {spec['hash']})")


if __name__ == "__main__":
    main()
