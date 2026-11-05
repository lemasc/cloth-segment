# Accessory class check (M6)

## In plain words
Four checks on the `accessory` class. Steps 1–3 need no GPU. Step 4 ran inference on the two saved U-Net checkpoints (GPU lock taken, no training).
DFMM eyeglass labels look clean, so label noise does not explain why `accessory` fails on DFMM.
DFMM has very little `accessory` data: 455 train images with it, and only 57 in val. Dropping `accessory` from the mean changes the run comparison from −0.0096 to −0.0008.
Per raw class (step 4): the retrain lost the sunglasses/eyeglass class (recall 0.0–0.6% against 24–41% in the original run), and scarf is almost never found in either run (0–4%). Glove is found on LIP only, at about 14–15% in both runs.
These numbers support the 11-class reporting that the T6 retrain report already asks for.

## 1. DFMM eyeglass masks
I looked at 24 random images (seed 0) of the 411 DFMM train images that contain raw class `eyeglass`, with the mask highlighted. Images are not committed (dataset licence).
- 22 show worn glasses, 2 show glasses hanging on a shirt collar. In all 24 the mask matches the glasses, lenses and frame.
- Mask size over the 411 images: median 1,845 pixels, median box 86 × 38 pixels (raw 750 × 1101 images). Mask fills 57% of its box (median). The rest of the box is mostly `face` (34%).
- Training resizes the long side to 512 (`configs/base.yaml`), so a median glasses mask is about 400 pixels (1,845 × (512/1101)²), a computed estimate, not a measured one.

What this says: the labels are not noisy in this sample. A sample of 24 cannot rule out rare bad masks.
What it does not say: why the model predicts `face` for 41–56% of these pixels. A small object (about 400 pixels) inside a large face region, with few examples, is a plausible cause, but I did not test it.

## 2. `accessory` supply per dataset and split (splits-v2 manifest)

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
From `eval/*/metrics.json` (original run eval-v1, retrain eval-v2). The 12-class value equals `miou` in each file.

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
- **The retrain lost sunglasses/eyeglass.** The original run found 24–41% of sunglasses pixels on LIP/CIHP and 27% of DFMM eyeglass pixels. The retrain finds almost none. Glove on LIP is the same in both runs (0.150 vs 0.137). So the 12-class gap between the two runs comes from sunglasses/eyeglass, not from the whole class.
- **Scarf is not learned in either run** (recall 0–4%). It is the biggest raw source of CIHP `accessory` pixels (1.91M of 2.82M test pixels, 68%), which caps CIHP `accessory` IoU. Glove on CIHP is also weak (1–2%), though it works somewhat on LIP.
- **Neither of the two classes you asked about is the problem.** DFMM gloves and tie have 0 test pixels. Glove on LIP is the only raw class both runs detect at the same level. This does not support dropping glove or tie.
- **Why scarf and sunglasses fail the way they do is not tested.** The "where the pixels go" numbers show what they are confused with, not why. 39% of LIP sunglasses pixels being predicted `headwear` was unexpected and I have not looked at it.

## What is still unknown
- Why LIP sunglasses are predicted `headwear` so often (view a few images).
- Whether more steps (80k) would make scarf or sunglasses stable. Only two 40k checkpoints were measured.
- Whether DFMM neckwear and tie masks are as clean as eyeglass. Not viewed.
- Seed-to-seed spread of each raw class: two runs only, and the original run had no seed on model init.

## Next
- Decide on `accessory` scope with this table, before T7/T8 (options: keep, split eyewear from clothing-like scarf, or ignore a class). Any label change is a `frozen-change`.
- Report 11-class and `accessory` separately in T9.
