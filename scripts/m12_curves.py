"""M12 training curves: val mIoU and train loss per check for the finished runs.

Reads each run's `log.jsonl` only (see `cloth_segment.curves`); writes the JSON,
the Markdown tables and one PNG named in the config (repo paths, committed).
No GPU, no dataset images. Never writes into `runs/` or `data/`.

Usage: uv run python scripts/m12_curves.py configs/report/m12.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cloth_segment import curves, paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", type=Path)
    a = p.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text())

    res = curves.build(cfg["runs"], paths.RUNS, cfg["datasets"])
    out = {k: paths.REPO_ROOT / v for k, v in cfg["out"].items()}
    written = [
        curves.write_json(res, out["json"]),
        out["tables"],
        curves.plot(res, out["figure"]),
    ]
    out["tables"].write_text(curves.markdown(res, cfg["datasets"]))
    for path in written:
        print("wrote", path.relative_to(paths.REPO_ROOT), f"({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
