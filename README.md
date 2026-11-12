# cloth-segment (submission code)

Core training and evaluation code for clothing-centric human parsing: U-Net
(ResNet-34), DeepLabV3+ (ResNet-50) and SegFormer-B2, plus a multi-task
SegFormer-B2 that also predicts DeepFashion-MultiModal garment attributes. All
models share one 12-class label set, one data pipeline, one training loop and
one evaluation protocol. No weights, datasets or results are included.

## Contents

| Path | What it is |
|---|---|
| `src/cloth_segment/data/` | Label set and converters (`labels.py`), dataset paths (`sources.py`), split manifest (`splits.py`), `Dataset` and sampler (`dataset.py`), augmentation (`transforms.py`) |
| `src/cloth_segment/models/` | The model registry and the four models |
| `src/cloth_segment/train.py` | Training loop, `python -m cloth_segment.train <config>` |
| `src/cloth_segment/evaluate.py` | Confusion matrix, mIoU / Dice, evaluation CLI |
| `src/cloth_segment/attributes.py`, `attr_eval.py` | Attribute loss, metrics and evaluation (multi-task model) |
| `src/cloth_segment/runs.py`, `hardware.py` | Config resolution and hash, run folders, GPU lock, GPU metadata |
| `configs/` | `base.yaml` (shared settings) and one small file per model |
| `tests/` | Unit tests; they need no datasets and no GPU |

## Setup

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

## Data layout

The code reads three folders in the repo root (create or symlink them):

- `datasets/`: raw LIP, CIHP and DeepFashion-MultiModal downloads.
- `data/`: converted masks (`data/masks/...`) and the split manifest
  (`data/splits/<version>/`). The version in `configs/base.yaml`
  (`data.version`) must match an existing folder.
- `runs/`: output, one folder per run.

The scripts that build `data/` (mask conversion, deduplication, splits) are
not part of this package. They live in the full project repository.

## Train and evaluate

```bash
uv run python -m cloth_segment.train configs/segformer_b2.yaml
uv run python -m cloth_segment.evaluate <run-folder>          # test split
uv run python -m cloth_segment.evaluate <run-folder> --split val
uv run python -m cloth_segment.attr_eval <run-folder>         # multi-task run only
```

Model configs (`unet_r34`, `deeplabv3p_r50`, `segformer_b2`, `segformer_b2_mt`)
only override the model; data, input size, augmentation and training budget
come from `base.yaml` and are shared. A finished run with the same config hash
is not trained again. Training uses a GPU when one is available.
