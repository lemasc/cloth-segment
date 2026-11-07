"""Pick images for a visual review and save each run's predicted masks. Needs the GPU lock.

Reads a review config (`configs/review/<task>.yaml`) and the per-image output
of `scripts/raw_class_breakdown.py` in `data/<breakdown_dir>/`. Writes to
`data/<out_dir>/`:

- `samples.csv`: one row per (question, image): how it was picked, its area
  and recall per run, and paths of the photo, shared mask and raw mask;
- `pred/<alias>/<dataset>/<index>.png`: predicted shared-class mask (uint8,
  original resolution) of each picked image, per run.

Every saved prediction is checked against the breakdown counts of the same
run (per raw class, pixels and hits). Existing files are never overwritten:
a different `samples.csv` stops the script, an existing mask is skipped.

Usage: uv run python scripts/review_predictions.py configs/review/m10.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset

from cloth_segment import paths, runs
from cloth_segment.evaluate import amp_dtype, load_eval_set, load_model, resolve_run
from cloth_segment.rawclass import (
    WithRawMask,
    iter_predictions,
    raw_collate,
    raw_pred_matrix,
    raw_rows,
)
from cloth_segment.review import load_review_config, per_image_recall, pick_samples, split_count


def read_breakdown(base: Path, run: str, ds: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = base / run / ds
    return pd.read_csv(d / "per_image_raw.csv"), pd.read_csv(d / "per_image_shared.csv")


def build_samples(cfg: dict) -> pd.DataFrame:
    base = paths.DATA / cfg["breakdown_dir"]
    rank = cfg["rank_run"]
    rows = []
    for qname, q in cfg["questions"].items():
        worst = split_count(q["n_worst"], len(q["datasets"]))
        rnd = split_count(q["n_random"], len(q["datasets"]))
        for ds, nw, nr in zip(q["datasets"], worst, rnd, strict=True):
            per_run = {}
            for alias, run in cfg["runs"].items():
                raw, shared = read_breakdown(base, run, ds)
                per_run[alias] = per_image_recall(raw, shared, q["raw"]).set_index("index")
            ref = per_run[rank]
            for alias, df in per_run.items():
                if not df["id"].equals(ref["id"]):
                    raise ValueError(f"{ds}: image order differs between {rank} and {alias}")
            pick = pick_samples(ref.reset_index(), nw, nr, q["min_share"], cfg["seed"])
            for alias, df in per_run.items():
                pick[f"recall_{alias}"] = df.loc[pick["index"], "recall"].to_numpy()
            rows.append(pick.assign(question=qname, dataset=ds))
    out = pd.concat(rows, ignore_index=True)
    out["key"] = out["dataset"] + ":" + out["index"].astype(str)
    return out


def add_paths(samples: pd.DataFrame, run_cfg: dict) -> pd.DataFrame:
    parts = []
    for ds, g in samples.groupby("dataset", sort=False):
        data = WithRawMask(load_eval_set(run_cfg, ds, "test"))
        r = data.dataset.rows
        idx = g["index"].to_numpy()
        if not (r.loc[idx, "id"].astype(str).to_numpy() == g["id"].astype(str).to_numpy()).all():
            raise ValueError(f"{ds}: breakdown ids do not match the test set")
        parts.append(
            g.assign(
                image=r.loc[idx, "image"].to_numpy(),
                mask=r.loc[idx, "mask"].to_numpy(),
                raw_mask=[str(Path(data.raw_paths[i]).relative_to(paths.DATASETS)) for i in idx],
            )
        )
    return pd.concat(parts).loc[samples.index]


def save_predictions(cfg: dict, samples: pd.DataFrame, device: torch.device) -> int:
    """Predict picked images per run; return the number of check failures."""
    base = paths.DATA / cfg["breakdown_dir"]
    out = paths.DATA / cfg["out_dir"] / "pred"
    failures = 0
    for alias, name in cfg["runs"].items():
        run_dir = resolve_run(name)
        run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        model, _ = load_model(run_dir, run_cfg, device)
        dtype = amp_dtype(run_cfg["train"]["amp"], device)
        for ds, g in samples.groupby("dataset", sort=False):
            raw_ref, _ = read_breakdown(base, name, ds)
            raw_ref = raw_ref.set_index(["index", "raw_name"])
            todo = sorted(
                i for i in g["index"].unique() if not (out / alias / ds / f"{i}.png").exists()
            )
            if not todo:
                continue
            data = WithRawMask(load_eval_set(run_cfg, ds, "test"))
            loader = DataLoader(
                Subset(data, todo),
                batch_size=run_cfg["train"]["batch_size"],
                shuffle=False,
                collate_fn=raw_collate,
            )
            (out / alias / ds).mkdir(parents=True, exist_ok=True)
            for idx, pred, _, raw in iter_predictions(model, loader, device, dtype):
                for r in raw_rows(raw_pred_matrix(pred, raw), ds):
                    ref = raw_ref.loc[(idx, r["raw_name"])]
                    if r["raw_px"] != ref["raw_px"] or (
                        r["hit_px"] is not None and r["hit_px"] != ref["hit_px"]
                    ):
                        failures += 1
                        print(f"MISMATCH {alias} {ds}:{idx} {r['raw_name']}", file=sys.stderr)
                arr = pred.to(torch.uint8).cpu().numpy()
                Image.fromarray(np.ascontiguousarray(arr), mode="L").save(
                    out / alias / ds / f"{idx}.png"
                )
            print(f"{alias} {ds}: saved {len(todo)} masks", flush=True)
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--wait", action="store_true", help="wait for the GPU lock")
    args = ap.parse_args(argv)
    cfg = load_review_config(args.config)
    rank_dir = resolve_run(cfg["runs"][cfg["rank_run"]])
    rank_cfg = yaml.safe_load((rank_dir / "config.yaml").read_text())

    samples = add_paths(build_samples(cfg), rank_cfg)
    cols = ["question", "dataset", "index", "id", "key", "pick", "raw_px", "hit_px", "share"]
    cols += [c for c in samples.columns if c.startswith("recall_")] + ["image", "mask", "raw_mask"]
    text = samples[cols].to_csv(index=False, lineterminator="\n")
    out = paths.DATA / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    target = out / "samples.csv"
    if target.exists() and target.read_text() != text:
        print(f"{target} exists and differs; use a new out_dir", file=sys.stderr)
        return 1
    target.write_text(text)
    print(f"{len(samples)} samples, {samples['key'].nunique()} images -> {target}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        with runs.gpu_lock(run="review-predictions", wait=args.wait):
            failures = save_predictions(cfg, samples, device)
    except runs.GpuBusy as e:
        print(e, file=sys.stderr)
        return 2
    if failures:
        print(f"{failures} raw-class counts differ from the breakdown", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
