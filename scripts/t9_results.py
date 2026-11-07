"""T9 results: tables, adjusted scores, bootstrap intervals and figures, from files only.

Reads each run's test `metrics.json` and the M10 breakdown under `data/`; writes
the JSON, the Markdown tables and the PNG figures named in the config (repo
paths, committed). No GPU, no dataset images. Never writes into `runs/` or `data/`.

Usage: uv run python scripts/t9_results.py configs/report/t9.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cloth_segment import paths, results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", type=Path)
    a = p.parse_args(argv)
    cfg = results.load_config(a.config)

    official, meta, raw_conf, counts = results.load_inputs(cfg, paths.RUNS, paths.DATA)
    res = results.build_results(cfg, official, meta, raw_conf, counts)

    out = {k: paths.REPO_ROOT / v for k, v in cfg["out"].items()}
    out["figures"].mkdir(parents=True, exist_ok=True)
    results.write_json(res, out["json"])
    out["tables"].write_text(results.tables_markdown(res))
    written = [out["json"], out["tables"]]
    written.append(results.plot_headline(res, out["figures"]))
    written.append(results.plot_differences(res, out["figures"]))
    written += [results.plot_per_class(res, out["figures"], m) for m in ("iou", "dice")]
    written += [results.plot_confusion(official, res, d, out["figures"]) for d in cfg["datasets"]]
    for path in written:
        print("wrote", path.relative_to(paths.REPO_ROOT), f"({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
