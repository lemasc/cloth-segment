"""Repo, data and run paths.

`datasets/`, `data/` and `runs/` are per-clone folders or symlinks (to shared
storage or local disk). Code should use these constants and never hard-code
absolute paths.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = REPO_ROOT / "datasets"  # raw downloads (read-only)
DATA = REPO_ROOT / "data"  # converted masks and the split manifest
RUNS = REPO_ROOT / "runs"  # one folder per training run
CONFIGS = REPO_ROOT / "configs"
