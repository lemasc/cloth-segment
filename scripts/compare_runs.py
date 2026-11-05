"""Compare two finished runs from their files: compare.md plus curve and lr PNGs per run.

Reads the run folders only (see `cloth_segment.compare`); writes only into OUT,
which should be outside the repo or not committed (no images in git).

Usage: uv run python scripts/compare_runs.py OLD_RUN NEW_RUN OUT [--exclude accessory]
       [--block 4000] [--shade-last 8000]
OLD_RUN / NEW_RUN: a run folder path or a run name under runs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cloth_segment import compare
from cloth_segment.evaluate import resolve_run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("old", help="reference run (path or name under runs/)")
    p.add_argument("new", help="run to compare against it")
    p.add_argument("out", type=Path, help="output folder (created if missing)")
    p.add_argument("--exclude", nargs="*", default=["accessory"], help="classes left out of mIoU")
    p.add_argument("--block", type=int, default=4000, help="steps per train-loss block")
    p.add_argument("--shade-last", type=int, default=8000, help="shade the last N steps in plots")
    a = p.parse_args(argv)

    old = compare.load_run(resolve_run(a.old))
    new = compare.load_run(resolve_run(a.new))
    a.out.mkdir(parents=True, exist_ok=True)
    md = a.out / "compare.md"
    md.write_text(compare.comparison_markdown(old, new, exclude=a.exclude, block=a.block))
    written = [md]
    for run in (old, new):
        written.append(compare.plot_curves(run, a.out, shade_last=a.shade_last))
        written.append(compare.plot_lr(run, a.out))
    for path in written:
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
