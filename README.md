# cloth-segment

Clothing-centric human parsing: pixel-level segmentation of garments and body
regions, trained on LIP, CIHP and DeepFashion-MultiModal mapped to one shared
label set. Three architectures (U-Net, DeepLabV3+, SegFormer) are compared.

Plan, scope and decisions: [`docs/PLAN.md`](docs/PLAN.md) (source of truth).
Session notes: [`docs/handoffs/`](docs/handoffs/).

## Setup

```bash
uv sync
scripts/setup_links.sh          # datasets/, data/, runs/ -> /shared/deeplearn/cloth
uv run pre-commit install       # ruff + safety hooks, once per clone
uv run pytest
```

## Layout

```
src/cloth_segment/
  data/        label set, converters, splits/manifest, Dataset, transforms
  models/      model builders
  runs.py      run naming, config hashing, locks, registry
  train.py     python -m cloth_segment.train <config.yaml>
  evaluate.py  python -m cloth_segment.evaluate <run> <test_set>
  paths.py     repo / shared-mount paths
configs/       base.yaml + one YAML per experiment
scripts/       CLI helpers (setup_links.sh, ...)
notebooks/
  eda/         marimo notebooks (uv run marimo edit notebooks/eda/<nb>.py)
  analysis/    Jupyter, outputs committed
tests/
docs/          PLAN.md, handoffs/, reports/
```

`datasets/`, `data/` and `runs/` are gitignored symlinks into the shared mount.

## Working conventions

Full rules: [`CONTRIBUTING.md`](CONTRIBUTING.md) (workflow, review, shared contracts),
[`docs/conventions.md`](docs/conventions.md) (code), [`AGENTS.md`](AGENTS.md) (coding agents).

- Commit before launching any run you might keep (the run name includes the commit).
- Long runs: launch from a terminal (tmux), not a notebook kernel.
- Mark a task `doing` in `docs/PLAN.md` when you start it, and push.
