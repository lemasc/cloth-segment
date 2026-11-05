# Accessory class check (M6)

## In plain words
Five checks on the `accessory` class. Steps 1–3 need no GPU (`scripts/accessory_data_check.py`). Steps 4 and 5 ran inference on the two saved U-Net checkpoints (GPU lock taken, no training).
DFMM eyeglass labels look clean in a sample of 24, so label noise is unlikely to explain why `accessory` fails on DFMM.
DFMM has very little `accessory` data: 455 train images with it, and only 57 in val. Dropping `accessory` from the mean changes the run comparison from −0.0096 to −0.0008.
Per raw class (step 4): the retrain lost almost all sunglasses/eyeglass hits (recall 0.0–0.6% against 24–41% in the original run). This is most, not all, of the gap: scarf also went from 4% to 0%. Glove is found on LIP only, at about 14–15% in both runs.
LIP sunglasses predicted as `headwear` (step 5): mostly images where the person also wears a hat or helmet, such as ski goggles on a helmet. Sunglasses on a bare face go to `face` in the retrain.
Decision: `accessory` stays as it is; the headline is mIoU without `accessory`, and `accessory` is used for error analysis (see "Decision and next").

## 1. DFMM eyeglass masks
Steps 1–3 script: `scripts/accessory_data_check.py` (helpers in `src/cloth_segment/rawclass.py`). Output: `docs/reports/2026-11-05-M6-accessory-data-check.json`. It reads only the manifest, the raw masks and the saved `metrics.json` files.

411 DFMM train images contain raw class `eyeglass`: 410 are 1101 × 750 pixels (height × width), 1 is 915 × 750.
- Visual check: 24 of them, drawn by the script with `--seed 0` (ids in the JSON), viewed as crops with the mask overlaid (`--montage-dir`). Images are not committed (dataset licence). 23 show glasses on the face and 1 shows glasses hanging on a shirt. In all 24 the mask matches the glasses, lenses and frame. All 24 have tinted lenses, so DFMM `eyeglass` in this sample is sunglasses.
  An earlier version of this check viewed a different sample of 24 from a scratch script that is lost; it found the same (22 on the face, 2 hanging, masks clean). The sample above replaces it.
- Mask size: median 1,845 pixels; median box 86 × 38 pixels (width × height). The mask fills a median 59% of its box (mean 57%). A median 34% of the box is `face`.
- At eval the long side is resized to `input.size` = 512 (`configs/base.yaml`), so a median mask is about 400 pixels at the model input (computed per image from the mask size, not measured on the model). Training also multiplies the long side by `augment.scale` (0.5–2.0), so in training the same mask is about 100–1,600 pixels.

What this says: the labels are not noisy in this sample. A sample of 24 cannot rule out rare bad masks.
What it does not say: why the model predicts `face` for 52–70% of DFMM eyeglass test pixels (step 4). A small object inside a large face region, with few examples, is a plausible cause, but I did not test it.

## 2. `accessory` supply per dataset and split (splits-v2 manifest)
Same script and JSON as step 1. "% of labelled pixels" leaves out IGNORE pixels.

| dataset | split | images | images with accessory | % of images | accessory pixels | % of labelled pixels |
|---|---|---|---|---|---|---|
| LIP | train | 25,653 | 3,404 | 13.3 | 8,743,432 | 0.374 |
| LIP | val | 1,358 | 199 | 14.7 | 769,951 | 0.622 |
| LIP | test | 9,632 | 1,892 | 19.6 | 2,895,180 | 0.410 |
| CIHP | train | 26,667 | 5,185 | 19.4 | 14,190,846 | 0.304 |
| CIHP | val | 1,420 | 272 | 19.2 | 775,126 | 0.312 |
| CIHP | test | 5,000 | 1,000 | 20.0 | 2,815,327 | 0.321 |
| DFMM | train | 10,180 | 455 | 4.5 | 1,423,771 | 0.017 |
| DFMM | val | 1,228 | 57 | 4.6 | 127,617 | 0.013 |
| DFMM | test | 1,293 | 72 | 5.6 | 157,182 | 0.015 |

- DFMM `accessory` is about 20 times rarer in pixels than in LIP/CIHP, but the M3 checkpoint-selection mean gives DFMM's `accessory` IoU the same weight (1/12 of 1/3) as every other class.
- The DFMM val `accessory` IoU rests on 57 images. LIP val has `accessory` at 0.622% of pixels against 0.374% in LIP train, so LIP val is also a noisy sample for this class.

## 3. 12-class and 11-class test mIoU
Same script and JSON as step 1, from `eval/*/metrics.json` (original run eval-v1, retrain eval-v2). The 12-class value equals `miou` in each file. The 11-class value is the mean of the other 11 class IoUs from the same file, so `accessory` errors on other classes' pixels still count against those classes.

| run | dataset | mIoU 12 classes | mIoU 11 (no accessory) | accessory IoU |
|---|---|---|---|---|
| original | LIP | 0.5539 | 0.5935 | 0.1182 |
| original | CIHP | 0.5756 | 0.6204 | 0.0825 |
| original | DFMM | 0.7736 | 0.8251 | 0.2077 |
| retrain | LIP | 0.5501 | 0.5924 | 0.0850 |
| retrain | CIHP | 0.5672 | 0.6185 | 0.0022 |
| retrain | DFMM | 0.7570 | 0.8257 | 0.0007 |

Mean of the three datasets: 12 classes 0.6344 (original) vs 0.6248 (retrain); 11 classes 0.6797 vs 0.6789; `accessory` alone 0.1361 vs 0.0293.

## 4. `accessory` per raw class (test sets, `best.pt` step 40000)
Script: `scripts/accessory_by_raw_class.py`; library: `src/cloth_segment/rawclass.py`. Raw data: `docs/reports/2026-11-05-M6-accessory-by-raw-class.json`.
It reuses the eval pipeline, so predictions are those behind the saved metrics. Cross-check: for all 6 run/dataset pairs, `accessory` ground-truth pixels and true positives equal the saved `confusion_matrix` row exactly.

*Recall* = pixels of the raw class predicted `accessory` / its pixels. *IoU\** = TP / (raw-class pixels + `accessory` predictions on non-accessory pixels); other raw accessory classes are left out, so a class is not punished for hits on a sibling. *Hit rate* = share of images with the class where at least half of its pixels are predicted `accessory`. Only recall and IoU\* are in the table; the rest is in the JSON.

| dataset | raw class | test pixels (images) | recall original | recall retrain | IoU\* original | IoU\* retrain |
|---|---|---|---|---|---|---|
| LIP | glove | 1,879,522 (1,310) | 0.150 | 0.137 | 0.126 | 0.126 |
| LIP | sunglasses | 319,498 (679) | 0.241 | 0.006 | 0.114 | 0.004 |
| LIP | scarf | 696,160 (197) | 0.037 | 0.001 | 0.025 | 0.001 |
| CIHP | glove | 522,334 (316) | 0.020 | 0.011 | 0.015 | 0.011 |
| CIHP | sunglasses | 379,419 (429) | 0.413 | 0.000 | 0.282 | 0.000 |
| CIHP | scarf | 1,913,574 (359) | 0.042 | 0.000 | 0.038 | 0.000 |
| DFMM | eyeglass | 124,247 (66) | 0.267 | 0.001 | 0.258 | 0.001 |
| DFMM | neckwear | 32,935 (6) | 0.011 | 0.000 | 0.010 | 0.000 |
| DFMM | gloves, tie | 0 | – | – | – | – |

Where the missed pixels go (largest shares, from the JSON; same in both runs unless noted):
- scarf: `outer` 37–47%, `upper` 22–31%. The model reads scarves as coat or top.
- sunglasses / eyeglass: `face` 25–70%, `headwear` up to 39% (LIP and CIHP), `hair` 6–13%.
- glove: `background` about 42–46%, then `outer`, `skin`.
- DFMM neckwear: `outer` 65–85%, `upper` 9–31%.

### What this shows
- **The gap between the runs is mostly sunglasses/eyeglass.** The original run found 24–41% of sunglasses pixels on LIP/CIHP and 27% of DFMM eyeglass pixels; the retrain finds almost none. Of the `accessory` true-positive pixels the retrain lost (from `tp_pixels` in the JSON), sunglasses/eyeglass are 61% on LIP, 65% on CIHP and 99% on DFMM. Scarf is 20% on LIP and 33% on CIHP: its small recall (about 4%) also went to 0. Glove is 19% on LIP, though its recall changed little (0.150 vs 0.137).
- **Scarf is not learned in either run** (recall 0–4%). It is the biggest raw source of CIHP `accessory` pixels (1.91M of 2.82M test pixels, 68%), which caps CIHP `accessory` IoU. Glove on CIHP is also weak (1–2%), though it works somewhat on LIP.
- **DFMM gloves and tie are not a factor:** they have 0 test pixels. Glove on LIP is the only raw class both runs find at a similar, non-zero level, so the data gives no reason to drop glove.
- **Why scarf and sunglasses fail the way they do is not tested.** The "where the pixels go" numbers show what they are confused with, not why. The 39% of LIP sunglasses pixels predicted `headwear` is looked at in step 5.

## 5. LIP sunglasses predicted as `headwear`
Question from step 4: 39% of LIP sunglasses pixels were predicted `headwear` by the retrain. Script: `scripts/lip_sunglasses_headwear.py`; summary: `docs/reports/2026-11-05-M6-lip-sunglasses-headwear.json`. It covers the 679 LIP test images with raw class `sunglasses` (median mask 227 pixels). Both runs, `best.pt` step 40000, GPU lock taken.

Where the sunglasses pixels go, split by whether the ground truth also has a `hat` in the image (shares of sunglasses pixels):

| run | group | images | accessory | headwear | face | hair | background |
|---|---|---|---|---|---|---|---|
| retrain | hat in GT | 352 | 0.010 | **0.560** | 0.278 | 0.013 | 0.100 |
| retrain | no hat in GT | 327 | 0.000 | 0.149 | **0.551** | 0.132 | 0.148 |
| original | hat in GT | 352 | 0.172 | 0.466 | 0.197 | 0.015 | 0.094 |
| original | no hat in GT | 327 | 0.340 | 0.062 | 0.323 | 0.132 | 0.115 |

- Over half of the images (352 of 679) also have a hat. In the retrain, 179 images have more than half their sunglasses pixels predicted `headwear`, and 153 of those (85%) have a hat in the ground truth.
- I viewed 12 montage images (4 with the most `headwear`, 8 from a seeded random sample; images not committed, licences). In all 4 of the worst cases the sunglasses are ski goggles or glasses on a helmet or hat, labelled `accessory` inside the head item. The retrain paints the whole head as `headwear`. The ground truth looked reasonable in all 12.
- On a bare face the retrain predicts `face` for most sunglasses pixels (55%), and the original run finds some of them (34% `accessory`).

What this says: the surprising `headwear` share is mostly a hat or helmet context effect, and the retrain's other main failure is small sunglasses absorbed into `face`. The original run finds sunglasses less often with a hat present (17% `accessory` against 34% without); the retrain finds almost none either way (1.0% and 0.01%).
What it does not say: whether this is caused by input resolution, class imbalance or the loss. The LIP labelling convention (goggles on helmets count as sunglasses) is only what the viewed images show, not a checked rule.

## What is still unknown
- Why small sunglasses are absorbed into `face` or `headwear` (resolution, loss and class weight are all untested).
- Whether more steps (80k) would make scarf or sunglasses stable. Only two 40k checkpoints were measured.
- Whether DFMM neckwear and tie masks are as clean as eyeglass. Not viewed.
- Seed-to-seed spread of each raw class: two runs only, and the original run had no seed on model init.

## Decision and next
Decided by lemasc (2026-11-05): `accessory` keeps its current mapping. The project compares architectures and makes no claim about accessory segmentation. So there is no label change and no `frozen-change`. Splitting eyewear from scarf would only be worth it if accessories became a claimed result, and then a second-seed run should come first to show whether eyewear is stable at 40k.

For T9:
- Headline test mIoU without `accessory` (11 classes), with the 12-class mIoU next to it. Report `accessory` IoU separately.
- Use `scripts/accessory_by_raw_class.py` on the T7/T8 runs for error analysis of `accessory`.
- Checkpoint selection still uses all 12 classes. For each model, check whether the 11-class val mean (per-class val IoU is in `log.jsonl` at every check) picks the same step as `best.pt`. If not, say so; the other checkpoint is not saved.
- A gap between models of about the size of the seed spread (about 0.01 in 12-class mIoU, from one pair of U-Net runs) cannot be told apart from seed.
