"""Repo and shared-mount paths.

`datasets/`, `data/` and `runs/` are per-clone symlinks into the shared mount,
created by `scripts/setup_links.sh`. Code should use these constants and never
hard-code `/shared/...`.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = REPO_ROOT / "datasets"  # raw downloads (read-only)
DATA = REPO_ROOT / "data"  # processed, versioned outputs of T2/T3
RUNS = REPO_ROOT / "runs"  # one folder per training run (D7)
CONFIGS = REPO_ROOT / "configs"
