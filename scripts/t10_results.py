"""T10 results: error zones, scores by image size and shape, CIHP dress review tally.

Reads the error-zone JSONs (`scripts/error_zones.py`), the M10 per-image counts
and the T10 review files; writes the JSON, Markdown tables and PNG figures named
in the config (repo paths, committed). No GPU, no dataset images.

Usage: uv run python scripts/t10_results.py configs/report/t10.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from cloth_segment import error_report, paths, results
from cloth_segment.review import load_review_config, read_verdicts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", type=Path)
    a = p.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text())
    aliases, datasets = list(cfg["runs"]), list(cfg["datasets"])

    zdir = paths.DATA / cfg["zones"]["out_dir"]
    bdir = paths.DATA / cfg["breakdown_dir"]
    zones = {
        al: {d: error_report.read_zones(zdir / r["name"] / f"{d}.json") for d in datasets}
        for al, r in cfg["runs"].items()
    }
    counts = {
        al: {
            d: results.read_image_counts(bdir / r["name"] / d / "per_image_shared.csv")
            for d in datasets
        }
        for al, r in cfg["runs"].items()
    }
    first = cfg["runs"][aliases[0]]["name"]
    sizes = {
        d: error_report.read_image_table(bdir / first / d / "per_image_shared.csv")
        for d in datasets
    }

    rcfg = load_review_config(paths.REPO_ROOT / cfg["review"])
    samples = pd.read_csv(paths.DATA / rcfg["out_dir"] / "samples.csv", dtype={"id": str})
    (q,) = rcfg["questions"].values()
    review = {
        "samples": samples,
        "verdicts": read_verdicts(paths.REPO_ROOT / rcfg["verdicts"]),
        "codes": q["verdicts"],
    }

    res = error_report.build(cfg, zones, counts, sizes, review)
    out = {k: paths.REPO_ROOT / v for k, v in cfg["out"].items()}
    out["figures"].mkdir(parents=True, exist_ok=True)
    results.write_json(res, out["json"])
    out["tables"].write_text(error_report.tables_markdown(res))
    written = [out["json"], out["tables"]]
    written.append(error_report.plot_zone_shares(res, out["figures"]))
    written.append(error_report.plot_class_zones(res, out["figures"]))
    written += [error_report.plot_buckets(res, k, out["figures"]) for k in cfg["buckets"]]
    for path in written:
        print("wrote", path.relative_to(paths.REPO_ROOT), f"({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
