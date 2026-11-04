# cloth-segment

Clothing-centric human parsing: pixel-level segmentation of garments and body
regions, trained on LIP, CIHP and DeepFashion-MultiModal mapped to one shared
label set. Three architectures (U-Net, DeepLabV3+, SegFormer) are compared.

Plan, scope and decisions: [`docs/PLAN.md`](docs/PLAN.md) (source of truth).
Session notes: [`docs/handoffs/`](docs/handoffs/).

## Setup

```bash
uv sync
# datasets/, data/, runs/ must exist: either your own folders/symlinks, or on the
# team machine:  scripts/setup_links.sh   (-> /shared/deeplearn/cloth)
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
  paths.py     repo / data / runs paths
configs/       base.yaml + one YAML per experiment
scripts/       CLI helpers (setup_links.sh, ...)
notebooks/
  eda/         marimo notebooks (uv run marimo edit notebooks/eda/<nb>.py)
  analysis/    Jupyter, outputs committed
tests/
docs/          PLAN.md, handoffs/, reports/
```

`datasets/`, `data/` and `runs/` are gitignored folders or symlinks (shared storage on the team machine, local disk elsewhere).

## Working conventions

Full rules: [`CONTRIBUTING.md`](CONTRIBUTING.md) (workflow, review, shared contracts),
[`docs/conventions.md`](docs/conventions.md) (code), [`AGENTS.md`](AGENTS.md) (coding agents).

- Commit before launching any run you might keep (the run name includes the commit).
- Long runs: launch from a terminal (tmux), not a notebook kernel. On a shared GPU, take the lock.
- Tell the team which task you take. Set it to `doing` in `docs/PLAN.md` as the first commit of your task branch, and `done` before review (same PR).
