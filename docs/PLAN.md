# Project Plan — Clothing-centric Human Parsing

> **Source of truth.** This file holds scope, decisions and task status only.
> Session notes, findings and work-in-progress go in `docs/handoffs/`.
> Update a task's status here when it changes. Do not add dates or schedules.

## Goal

Pixel-level clothing segmentation ("human parsing" with a clothing-centric
label set): predict a semantic mask over a shared set of classes (about 12:
background, body regions, garment types, accessories), normalised across
several public datasets. Compare 3 architectures fairly, from CNN to
Transformer, and evaluate on held-out test sets with mIoU, per-class IoU and
Dice, plus error analysis.

Rubric: `docs/assignment-rubric.md` (segmentation = 3 pts difficulty;
multi-task = 4 pts, see E1).

## Scope

| In | Out (for now) |
|---|---|
| LIP, CIHP (in-the-wild), DeepFashion-MultiModal (studio fashion, incl. attribute labels) | ATR, Fashionpedia, DeepFashion2 / DressCode (pending access) |

## Decisions

| ID | Decision |
|---|---|
| D1 | Train on LIP + CIHP + DFMM, all mapped to one shared label set. |
| D2 | ATR excluded for now (no coat/outer class, which conflicts with the shared set). |
| D3 | The data loader also loads DFMM attribute labels from the start, so multi-task (E1) needs no loader rework. |
| D4 | Official LIP/CIHP val sets serve as test sets (official test labels are unavailable). They get the same exclusion rules as train: images with annotation failures (background-only masks) are dropped from every split. The rules, thresholds and per-split counts are recorded with the splits version. DFMM is split by outfit ID to prevent leakage. Train images that share a source photo or are near-duplicates of a test image are dropped. The test sets get no other change. |
| D5 | Fair comparison: same data, splits, input resolution, augmentation, iteration budget and pretrained (ImageNet) encoders across all models. |
| D6 | Python source for library, training and evaluation; one YAML config per experiment. Marimo for EDA; Jupyter (outputs committed) for analysis. Long runs are launched from a terminal, not a notebook kernel. |
| D7 | Runs live on the shared mount, one folder each, named `<datetime>-<commit>-<slug>` and set at run start. The resolved config hash (+ data version) in `meta.json` is the cache key: a finished run with the same hash is never retrained. |
| D8 | Preprocessing tiers. Dataset construction and evaluation protocol are versioned shared contracts, not experiments. Each is frozen once its freeze condition is met (status table in `CONTRIBUTING.md` section 4); until then changes are allowed but must be declared and version-bumped. Training-time input (resolution, augmentation, sampling) is config, so it is part of the run hash. |
| D9 | Shared machine: one Linux user and git clone per member; data and runs on a shared group-writable mount; one GPU, so training needs a GPU lock. |

## Tasks

Status: `todo` · `doing` · `done` · `blocked` · `dropped`

### Core (minimum requirements)

| ID | Task | Status |
|---|---|---|
| T1 | EDA of LIP, CIHP, DFMM (sizes, class distribution, samples, label conflicts) | done |
| T2 | Shared label set and per-dataset converters | done |
| T3 | Splits and manifest (group split for DFMM, duplicate check across datasets) | done |
| T4 | Data loader (segmentation + DFMM attributes) and augmentation | todo |
| T5 | Training and evaluation pipeline (config-driven, shared by all models) | todo |
| T6 | Model A: U-Net (ResNet-34) | doing |
| T7 | Model B: DeepLabV3+ (ResNet-50) | todo |
| T8 | Model C: SegFormer-B2 | todo |
| T9 | Evaluation: mIoU / per-class IoU / Dice per test set, confusion matrix, qualitative comparisons | todo |
| T10 | Error analysis and limitations | todo |
| T11 | Slides and README (usage, data sources, configuration) | todo |

### Extensions (if time allows)

| ID | Task | Status |
|---|---|---|
| E1 | Multi-task on DFMM: segmentation + attribute heads, reported per task (for the 4-pt difficulty score) | todo |
| E2 | Single-dataset vs merged training; cross-dataset table | todo |
| E3 | Loss comparison: CE vs CE + Dice | todo |
| E4 | Fashionpedia as out-of-domain test set (garment classes only) | todo |
| E5 | Revisit ATR | todo |

### Maintenance (decisions and data fixes, not deliverables)

| ID | Task | Status |
|---|---|---|
| M1 | D4 empty-mask rule for test sets (PR #3) | done |
| M2 | Size and shape filter for LIP/CIHP/DFMM, `splits-v2` | done |
