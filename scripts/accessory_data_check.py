"""Steps 1-3 of the M6 `accessory` check. No GPU, no model; reads data/ and runs/ only.

1. DFMM eyeglass masks in the train split: size, bounding box, how much of the
   box the mask fills, and the share of the box that is `face`. Optionally
   writes montages of a seeded random sample for a visual check (images are
   not for committing: dataset licence).
2. `accessory` images and pixels per dataset and split (splits manifest).
3. 12-class mIoU, mIoU without `accessory`, and `accessory` IoU per test set,
   from each run's saved `metrics.json` (newest eval version that has one).

Usage: uv run python scripts/accessory_data_check.py RUN [RUN ...] --out FILE.json
           [--montage-dir DIR] [--n-sample 24] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from cloth_segment.data.sources import RAW_LABELS, list_pairs
from cloth_segment.data.splits import load_manifest
from cloth_segment.evaluate import TEST_SETS, resolve_run
from cloth_segment.rawclass import accessory_supply, box_stats, miou_without

RAW_ID = {v: k for k, v in RAW_LABELS["dfmm"].items()}
EYEGLASS, FACE = RAW_ID["eyeglass"], RAW_ID["face"]


def eyeglass_stats(manifest, input_size: int) -> tuple[dict, dict]:
    """Summary over DFMM train images with `eyeglass`, and {id: (pair, stats)}."""
    rows = manifest[(manifest["dataset"] == "dfmm") & (manifest["split"] == "train")]
    pairs = {}
    for split in rows["source_split"].unique():
        pairs |= {(split, p.id): p for p in list_pairs("dfmm", split)}
    found = {}
    for split, i in zip(rows["source_split"], rows["id"], strict=True):
        pair = pairs[(split, i)]
        with Image.open(pair.mask) as im:
            s = box_stats(np.asarray(im), EYEGLASS, FACE)
        if s is not None:
            found[i] = (pair, s)
    stats = [s for _, s in found.values()]
    sizes: dict[str, int] = {}
    for s in stats:
        key = f"{s['image_hw'][0]}x{s['image_hw'][1]}"
        sizes[key] = sizes.get(key, 0) + 1
    # eval resizes the long side to input.size; training scales that by augment.scale
    at_eval = [s["pixels"] * (input_size / max(s["image_hw"])) ** 2 for s in stats]

    def med(key: str) -> float:
        return float(np.median([s[key] for s in stats]))

    summary = {
        "images": len(stats),
        "image_sizes_hw": sizes,
        "median_pixels": med("pixels"),
        "median_box_w": med("box_w"),
        "median_box_h": med("box_h"),
        "median_fill": med("fill"),
        "mean_fill": float(np.mean([s["fill"] for s in stats])),
        "median_face_share_of_box": med("around_share"),
        "median_pixels_at_eval_input": float(np.median(at_eval)),
    }
    return summary, found


def montage(picks: list[tuple[str, object, dict]], out_dir: Path, per_file: int = 8) -> None:
    """Per sample: crop around the glasses, and the same crop with the mask in red."""
    h = 200
    tiles = []
    for n, (_, pair, s) in enumerate(picks):
        with Image.open(pair.image) as im:
            img = np.asarray(im.convert("RGB"))
        with Image.open(pair.mask) as im:
            sel = np.asarray(im) == EYEGLASS
        y0, y1, x0, x1 = s["box"]
        side = max(3 * max(y1 - y0, x1 - x0), 160)
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        top, left = max(0, cy - side // 2), max(0, cx - side // 2)
        crop = img[top : top + side, left : left + side]
        over = crop.copy()
        m = sel[top : top + side, left : left + side]
        over[m] = (0.5 * over[m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        pair_tile = Image.fromarray(np.concatenate([crop, over], axis=1))
        pair_tile = pair_tile.resize((int(pair_tile.width * h / pair_tile.height), h))
        ImageDraw.Draw(pair_tile).text((4, 4), str(n), fill=(255, 255, 0))
        tiles.append(np.asarray(pair_tile))
    width = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 4), (0, width - t.shape[1]), (0, 0))) for t in tiles]
    out_dir.mkdir(parents=True, exist_ok=True)
    for part in range(0, len(tiles), per_file):
        chunk = tiles[part : part + per_file]
        Image.fromarray(np.concatenate(chunk, axis=0)).save(
            out_dir / f"eyeglass_sample_{part // per_file}.png"
        )


def saved_metrics(run_dir: Path, dataset: str) -> Path:
    """Test metrics of the newest eval version that scored `dataset`."""
    found = sorted(run_dir.glob(f"eval/eval-v*/{dataset}/metrics.json"))
    if not found:
        raise FileNotFoundError(f"{run_dir.name}: no test metrics for {dataset}")
    return found[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="finished runs for step 3")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--montage-dir", type=Path, help="write step 1 sample montages here")
    ap.add_argument("--n-sample", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    manifest, meta = load_manifest()
    run_dirs = [resolve_run(r) for r in args.runs]
    cfg = yaml.safe_load((run_dirs[0] / "config.yaml").read_text())
    summary, found = eyeglass_stats(manifest, cfg["input"]["size"])
    ids = random.Random(args.seed).sample(sorted(found), min(args.n_sample, len(found)))
    summary["sample"] = {"seed": args.seed, "ids": ids}
    if args.montage_dir:
        montage([(i, *found[i]) for i in ids], args.montage_dir)

    supply = accessory_supply(manifest)
    scores = {}
    for run_dir in run_dirs:
        scores[run_dir.name] = {}
        for ds in TEST_SETS:
            path = saved_metrics(run_dir, ds)
            scores[run_dir.name][ds] = miou_without(json.loads(path.read_text())) | {
                "metrics_file": str(path.relative_to(run_dir))
            }

    result = {
        "data_version": meta["data_version"],
        "step1_dfmm_eyeglass_train": summary,
        "step2_accessory_supply": supply.to_dict(orient="records"),
        "step3_test_miou": scores,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
