# cloth-segment

Clothing-centric human parsing: predict a class for every pixel of a photo of a
person (background, hair, face, skin, 6 garment types, headwear, accessory).
Three public datasets (LIP, CIHP, DeepFashion-MultiModal) are mapped to one
shared 12-class label set. Three architectures, from CNN to Transformer
(U-Net, DeepLabV3+, SegFormer), are compared under the same data, splits,
augmentation and training budget. A multi-task SegFormer also predicts 12
garment attributes from the same encoder.

**Start here:** [`docs/reports/2026-11-09-M12-final-report.md`](docs/reports/2026-11-09-M12-final-report.md)
is the full write-up (Thai, English terms): data, method, implementation,
results, error analysis, limitations, and an index of every task's handoff and
report. Scope, decisions (D1–D9) and task status live in [`docs/PLAN.md`](docs/PLAN.md).

## Results

Test mIoU over 11 classes (without `accessory`, see [M6](docs/handoffs/2026-11-05-M6-lemasc.md)),
with 95% bootstrap intervals over test images. 40k steps, one run per model.

| test set | images | U-Net R34 (X2) | DeepLabV3+ R50 (X3) | SegFormer-B2 (X4) |
|---|---|---|---|---|
| LIP | 9,632 | 0.5924 (0.585–0.600) | 0.6071 (0.600–0.615) | **0.6399** (0.632–0.647) |
| CIHP | 5,000 | 0.6185 (0.611–0.625) | 0.6262 (0.619–0.633) | **0.6682** (0.660–0.676) |
| DFMM | 1,293 | 0.8257 (0.815–0.837) | 0.8245 (0.813–0.836) | **0.8635** (0.852–0.875) |
| mean of test sets | – | 0.6789 | 0.6859 | **0.7239** |

- SegFormer-B2 is best on every test set and every class. The gap to the other
  two (about +0.04) is far larger than the test-image uncertainty and than the
  difference between two SegFormer seeds (≤ 0.002).
- DeepLabV3+ leads U-Net by only +0.007; one seed each, so this may not hold.
- **Multi-task (X5):** SegFormer-B2 with 12 attribute heads reaches attribute
  macro-F1 **0.8036** on DFMM test (always guessing the most common value: 0.3310), and
  its segmentation is within 0.0009 mIoU of the same model without heads (X6).
- All models were still improving at 40k steps: this is an equal-budget
  comparison, not each model's best score.

Per-class IoU/Dice, confusion matrices, error analysis and limitations:
sections 5–7 of the [final report](docs/reports/2026-11-09-M12-final-report.md).

## Data

| Dataset | Content | Source |
|---|---|---|
| LIP | one-person crops from everyday photos | <https://github.com/Yukun-Huang/Single-Human-Parsing-LIP> |
| CIHP | everyday photos, several people per image | <https://github.com/Engineering-Course/CIHP_PGN> |
| DeepFashion-MultiModal (DFMM) | studio fashion photos, parsing masks and attribute labels | <https://github.com/yumingj/DeepFashion-MultiModal> |

Download each dataset from its source (each has its own licence and terms of
use) and place it under `datasets/` as follows:

```
datasets/
  lip/                     TrainVal_images/, TrainVal_parsing_annotations/
  cihp/                    Training/, Validation/  (Images/, Category_ids/)
  deepfashion-multimodal/  images/, segm/, labels/
```

- The official LIP and CIHP test sets have no labels, so their official val
  sets are our test sets. DFMM is split by outfit ID.
- Near-duplicates across and within datasets are found with an image hash and
  removed from train when they match a test image.
- Final split (`splits-v2`): 62,500 train / 4,006 val / 15,925 test images.

Details: section 2 of the [final report](docs/reports/2026-11-09-M12-final-report.md),
[T2](docs/handoffs/2026-11-03-T2-lemasc.md) (label set), [T3](docs/handoffs/2026-11-03-T3-lemasc.md) (splits).

## Setup

Python 3.12 and [`uv`](https://docs.astral.sh/uv/). Training used one 12 GB
GPU (RTX 4070); batch 8 at 512×512 with bf16 fits in that memory.

```bash
uv sync
# datasets/, data/ and runs/ must exist: your own folders or symlinks, or on the
# team machine:
scripts/setup_links.sh          # -> /shared/deeplearn/cloth
uv run pre-commit install       # ruff + safety hooks, once per clone
uv run pytest                   # no datasets or GPU needed
```

## Usage

```bash
# 1. Data, once. Outputs are versioned and never overwritten.
uv run python scripts/prepare_masks.py     # raw masks -> 12 shared classes, data/masks/labels-v1/
uv run python scripts/build_splits.py      # splits + duplicate check, data/splits/splits-v2/

# 2. Train (takes the GPU lock; skipped if a finished run has the same config hash)
uv run python -m cloth_segment.train configs/segformer_b2.yaml

# 3. Evaluate the best checkpoint on the test sets
uv run python -m cloth_segment.evaluate <run-folder> --wait
uv run python -m cloth_segment.attr_eval <run-folder> --wait   # multi-task runs only
```

Each run writes `runs/<datetime>-<commit>-<slug>/` with `config.yaml`,
`meta.json` (commit, config hash, data version, GPU), `log.jsonl`
(train loss and val scores), `ckpt/` and `eval/eval-v2/<test set>/metrics.json`.
Checkpoints are not in git.

Commands to rebuild every table and figure in the reports: section 9 of the
[final report](docs/reports/2026-11-09-M12-final-report.md).

## Configuration

Every experiment is one YAML file in `configs/`, merged over
[`configs/base.yaml`](configs/base.yaml). The merged config is hashed, and the
hash names the run.

- `base.yaml` holds the fields shared by every model for a fair comparison
  (D5): data version and sampling, input size 512, augmentation, batch 8,
  40,000 steps, AdamW (lr 1e-4, warmup then linear decay to 0), cross-entropy,
  bf16, checkpoint rule.
- Experiment configs change only `model.*` (and the seed where stated):

| Config | Model | Run |
|---|---|---|
| [`unet_r34.yaml`](configs/unet_r34.yaml) | U-Net, ResNet-34 encoder | X2 |
| [`deeplabv3p_r50.yaml`](configs/deeplabv3p_r50.yaml) | DeepLabV3+, ResNet-50 encoder | X3 |
| [`segformer_b2.yaml`](configs/segformer_b2.yaml) | SegFormer-B2 (MiT-B2), seed 0 | X4 |
| [`segformer_b2_s1.yaml`](configs/segformer_b2_s1.yaml) | SegFormer-B2, seed 1 | X6 |
| [`segformer_b2_mt.yaml`](configs/segformer_b2_mt.yaml) | SegFormer-B2 + 12 attribute heads, seed 1 | X5 |

All encoders are ImageNet-pretrained and built with
`segmentation_models_pytorch`. `configs/report/` and `configs/review/` drive the
analysis scripts; they do not train anything.

## Layout

```
src/cloth_segment/
  data/          label set and converters, splits and duplicate check, Dataset, transforms
  models/        U-Net, DeepLabV3+, SegFormer, multi-task SegFormer (one registry)
  train.py       python -m cloth_segment.train <config.yaml>
  evaluate.py    python -m cloth_segment.evaluate <run> [test sets]
  attr_eval.py   attribute scores for multi-task runs
  attributes.py  DFMM attribute heads: targets, loss, metrics
  runs.py        run naming, config hashing, GPU lock
  results.py, errors.py, error_report.py, multiperson.py, curves.py   report analysis
configs/         base.yaml + one YAML per experiment; report/ and review/ for analysis
scripts/         data preparation and report scripts (thin wrappers)
notebooks/
  eda/           marimo notebooks (uv run marimo edit notebooks/eda/<nb>.py)
  review/        marimo, visual review of images, labels and predictions
tests/           runs without datasets or a GPU
docs/            PLAN.md, reports/, handoffs/
```

`datasets/`, `data/` and `runs/` are gitignored folders or symlinks (shared
storage on the team machine, local disk elsewhere).

## Documentation

| Where | What |
|---|---|
| [`docs/reports/2026-11-09-M12-final-report.md`](docs/reports/2026-11-09-M12-final-report.md) | Final report, start here (Thai) |
| [`docs/PLAN.md`](docs/PLAN.md) | Scope, decisions D1–D9, task status (source of truth) |
| [`docs/reports/`](docs/reports/) | One report per evaluation or analysis (T9, T10, M10, M11, X2–X6) |
| [`docs/handoffs/`](docs/handoffs/) | Session notes per task: what was done, why, what is open |
| [`docs/assignment-rubric.md`](docs/assignment-rubric.md) | Course rubric |

## Working conventions

Full rules: [`CONTRIBUTING.md`](CONTRIBUTING.md) (workflow, review, shared contracts),
[`docs/conventions.md`](docs/conventions.md) (code), [`AGENTS.md`](AGENTS.md) (coding agents).

- Commit before launching any run you might keep (the run name includes the commit).
- Long runs: launch from a terminal (tmux), not a notebook kernel. On a shared GPU, take the lock.
- Tell the team which task you take. Set it to `doing` in `docs/PLAN.md` as the first commit of your task branch, and `done` before review (same PR).
