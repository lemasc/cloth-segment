"""Scan every labelled image/mask pair once and cache per-pair stats for EDA (T1).

Decodes every image and mask fully, so corrupt files are caught. Output: one
CSV per dataset in `data/eda/`, one row per pair:

    dataset, split, id, img_w, img_h, img_mode, exif_orient, mask_w, mask_h,
    mask_mode, error, unexpected, v0..vN (pixel count per raw label value)

Usage: uv run python scripts/eda_scan.py [lip cihp dfmm] [--workers 16]
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from cloth_segment.data.sources import RAW_LABELS, SPLITS, Pair, list_pairs
from cloth_segment.paths import DATA

OUT = DATA / "eda"
EXIF_ORIENTATION = 0x0112


def scan_pair(args: tuple[Pair, int]) -> dict:
    pair, n_values = args
    row = {"dataset": pair.dataset, "split": pair.split, "id": pair.id, "error": "", "unexpected": ""}
    try:
        with Image.open(pair.image) as im:
            row.update(img_w=im.width, img_h=im.height, img_mode=im.mode,
                       exif_orient=im.getexif().get(EXIF_ORIENTATION, ""))
            im.load()
        with Image.open(pair.mask) as m:
            row.update(mask_w=m.width, mask_h=m.height, mask_mode=m.mode)
            arr = np.asarray(m)
        if arr.ndim != 2:
            row["error"] = f"mask ndim {arr.ndim}"
            return row
        counts = np.bincount(arr.ravel(), minlength=256)
        row.update({f"v{v}": int(counts[v]) for v in range(n_values)})
        extra = np.flatnonzero(counts[n_values:]) + n_values
        row["unexpected"] = " ".join(f"{v}:{counts[v]}" for v in extra)
    except Exception as e:  # noqa: BLE001 - record and keep scanning
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def scan_dataset(dataset: str, workers: int) -> None:
    n_values = max(RAW_LABELS[dataset]) + 1
    pairs = [p for s in SPLITS[dataset] for p in list_pairs(dataset, s)]
    cols = ["dataset", "split", "id", "img_w", "img_h", "img_mode", "exif_orient",
            "mask_w", "mask_h", "mask_mode", "error", "unexpected"] + [f"v{v}" for v in range(n_values)]
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f".scan_{dataset}.csv.tmp"
    with ProcessPoolExecutor(workers) as ex, tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for i, row in enumerate(ex.map(scan_pair, ((p, n_values) for p in pairs), chunksize=64)):
            w.writerow(row)
            if (i + 1) % 5000 == 0:
                print(f"  {dataset}: {i + 1}/{len(pairs)}", flush=True)
    tmp.replace(OUT / f"scan_{dataset}.csv")
    print(f"{dataset}: {len(pairs)} pairs -> {OUT / f'scan_{dataset}.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(SPLITS))
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    a = ap.parse_args()
    for d in a.datasets:
        scan_dataset(d, a.workers)


if __name__ == "__main__":
    main()
