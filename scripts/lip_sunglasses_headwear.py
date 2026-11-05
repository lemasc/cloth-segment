"""Where do LIP test sunglasses pixels go, and does a hat in the image matter? (M6)

For every LIP test image with raw class `sunglasses`, records per run the share
of its sunglasses pixels predicted as each shared class and whether the ground
truth also has a `hat`. Writes per-image CSVs, a summary JSON and montages
(image, ground truth, one panel per run) for a seeded random sample and for the
images with the most `headwear` predictions (by the first run).
Needs the GPU lock. Read-only on runs/ and data/.

Usage: uv run python scripts/lip_sunglasses_headwear.py RUN [RUN ...] --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from cloth_segment import runs
from cloth_segment.data.dataset import SegDataset
from cloth_segment.data.labels import PALETTE, SHARED_CLASSES, convert
from cloth_segment.data.sources import RAW_LABELS, list_pairs
from cloth_segment.evaluate import amp_dtype, load_eval_set, load_model, resolve_run
from cloth_segment.rawclass import WithRawMask, headwear_summary, iter_predictions, raw_loader

RAW_ID = {v: k for k, v in RAW_LABELS["lip"].items()}
SUN, HAT = RAW_ID["sunglasses"], RAW_ID["hat"]
TABLE = np.array([*PALETTE, *[(255, 255, 255)] * (256 - len(PALETTE))], dtype=np.uint8)


def sunglasses_subset(cfg: dict) -> SegDataset:
    """LIP test images whose raw mask contains `sunglasses`."""
    full = load_eval_set(cfg, "lip", "test")
    paths = WithRawMask(full).raw_paths
    keep = [i for i, p in enumerate(paths) if (np.asarray(Image.open(p)) == SUN).any()]
    return SegDataset(full.rows.iloc[keep], full.size, train=False, augment=full.augment)


def run_images(run_dir: Path, device: torch.device, subset: SegDataset | None):
    """Per-image rows and uint8 predictions of one run on the sunglasses images."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model, _ = load_model(run_dir, cfg, device)
    subset = subset or sunglasses_subset(cfg)
    ids = subset.rows["id"].tolist()
    workers = (cfg.get("run") or {}).get("num_workers", 0)
    loader = raw_loader(subset, cfg["train"]["batch_size"], workers)
    rows, preds = [], {}
    for i, pred, _target, raw in iter_predictions(
        model, loader, device, amp_dtype(cfg["train"]["amp"], device)
    ):
        d = torch.bincount(pred[raw == SUN].long(), minlength=len(SHARED_CLASSES)).cpu()
        n = int(d.sum())
        rows.append(
            {"id": ids[i], "sunglasses_px": n, "hat_px_gt": int((raw == HAT).sum())}
            | {f"pred_{c}": int(d[k]) / n for k, c in enumerate(SHARED_CLASSES)}
        )
        preds[ids[i]] = pred.to("cpu", torch.uint8).numpy()
    return subset, pd.DataFrame(rows).set_index("id"), preds


def montage(ids: list[str], preds_by_run: dict[str, dict], out: Path) -> None:
    """Rows of [image | ground truth | prediction per run], 4 images per file."""
    lookup = {p.id: p for p in list_pairs("lip", "val")}
    tiles = []
    for i in ids:
        raw = np.asarray(Image.open(lookup[i].mask))
        panels = [np.asarray(Image.open(lookup[i].image).convert("RGB"))]
        panels += [TABLE[convert(raw, "lip")]] + [TABLE[p[i]] for p in preds_by_run.values()]
        h = 300
        panels = [
            np.asarray(Image.fromarray(p).resize((max(1, int(p.shape[1] * h / p.shape[0])), h)))
            for p in panels
        ]
        tiles.append(np.concatenate(panels, axis=1))
    width = max(t.shape[1] for t in tiles)
    tiles = [
        np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0)), constant_values=255) for t in tiles
    ]
    for part in range(0, len(tiles), 4):
        Image.fromarray(np.concatenate(tiles[part : part + 4], axis=0)).save(
            out.with_name(f"{out.name}_{part // 4}.png")
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-random", type=int, default=12)
    ap.add_argument("--n-worst", type=int, default=12)
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames, preds, subset = {}, {}, None
    try:
        with runs.gpu_lock(run="lip-sunglasses-headwear"):
            for name in args.runs:
                run_dir = resolve_run(name)
                subset, frames[run_dir.name], preds[run_dir.name] = run_images(
                    run_dir, device, subset
                )
                print(f"{run_dir.name}: {len(subset)} images done", flush=True)
    except runs.GpuBusy as e:
        print(e, file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(args.out_dir / f"per_image_{name}.csv")
    summary = {name: headwear_summary(df) for name, df in frames.items()}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    first = next(iter(frames.values()))
    picks = {
        "random": random.Random(0).sample(sorted(first.index), args.n_random),
        "worst_headwear": first.sort_values("pred_headwear", ascending=False)
        .index[: args.n_worst]
        .tolist(),
    }
    for tag, ids in picks.items():
        montage(ids, preds, args.out_dir / f"montage_{tag}")
        (args.out_dir / f"montage_{tag}_ids.json").write_text(json.dumps(ids) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
