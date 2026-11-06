"""Fill missing GPU fields in the meta.json of old runs (M9). Run on the machine that trained them.

Only adds fields that are absent, never overwrites, and refuses runs from another host.
The values come from this machine's GPU and driver now, so the driver may be newer than
the one the run used; touched runs get `hw_backfilled: true`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cloth_segment.hardware import BackfillRefused, backfill_run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", type=Path, help="run folders (each has a meta.json)")
    p.add_argument(
        "--dry-run", action="store_true", help="print the planned changes, write nothing"
    )
    args = p.parse_args(argv)
    failed = 0
    for run_dir in args.runs:
        try:
            changes = backfill_run(run_dir, dry_run=args.dry_run)
        except (BackfillRefused, OSError, ValueError) as e:
            print(f"refused {run_dir}: {e}", file=sys.stderr)
            failed += 1
            continue
        verb = "would set" if args.dry_run else "set"
        print(f"{run_dir}: {verb} {changes}" if changes else f"{run_dir}: nothing to do")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
