"""T9 qualitative grids: photo, ground truth and each run's prediction, one JPEG per test set.

Reads the sample list and predicted masks that `scripts/review_predictions.py`
saved under `data/<out_dir>/` for the same config, plus the test photos and
shared masks. No GPU. Writes the grids to the config's `figures` folder under
`data/` (shared mount): they contain dataset photos, so they stay out of git.
Each prediction panel shows that image's mIoU without the excluded classes
(same rule as `evaluate`, one image only).

Usage: uv run python scripts/t9_qualitative.py configs/review/t9.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cloth_segment import paths
from cloth_segment.review import (
    colour_mask,
    comparison_grid,
    image_miou,
    load_review_config,
)

EXCLUDE = ("accessory",)  # headline mIoU rule (M6), as in configs/report/t9.yaml


def _wrap(text: str, width: int = 18) -> str:
    """Break a long image id over lines so it fits the label column."""
    return "\n".join(text[i : i + width] for i in range(0, len(text), width))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path)
    args = ap.parse_args(argv)
    cfg = load_review_config(args.config)
    base = paths.DATA / cfg["out_dir"]
    samples = pd.read_csv(base / "samples.csv", dtype={"id": str})
    out = paths.DATA / cfg["figures"]
    out.mkdir(parents=True, exist_ok=True)
    aliases = list(cfg["runs"])

    for ds, g in samples.groupby("dataset", sort=False):
        rows = []
        for r in g.itertuples():
            photo = np.asarray(Image.open(paths.DATASETS / r.image).convert("RGB"))
            gt = np.asarray(Image.open(paths.DATA / r.mask))
            panels, captions = [photo, colour_mask(gt)], ["", ""]
            for a in aliases:
                pred = np.asarray(Image.open(base / "pred" / a / ds / f"{r.index}.png"))
                m = image_miou(pred, gt, EXCLUDE)
                panels.append(colour_mask(pred))
                captions.append("-" if m is None else f"mIoU {m:.2f}")
            rows.append(
                {
                    "label": f"{r.pick}\n{ds} #{r.index}\n{_wrap(r.id)}",
                    "panels": panels,
                    "captions": captions,
                }
            )
        grid = comparison_grid(
            rows,
            ["photo", "ground truth", *[f"{a} {cfg['run_models'][a]}" for a in aliases]],
            int(cfg["panel_height"]),
            note=(
                f"{ds.upper()} test. random: drawn with seed {cfg['seed']}; worst: lowest garment "
                f"recall of {cfg['rank_run']}. mIoU per image without {', '.join(EXCLUDE)}; "
                "grey = ignored pixels."
            ),
        )
        path = out / f"qualitative_{ds}.jpg"
        grid.save(path, quality=int(cfg["jpeg_quality"]), optimize=True)
        print("wrote", path.relative_to(paths.DATA), f"({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
