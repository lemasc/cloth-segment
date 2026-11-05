# Accessory class check (M6)

## In plain words
Three cheap checks on the `accessory` class, no training and no GPU.
DFMM eyeglass labels look clean, so label noise does not explain why `accessory` fails on DFMM.
DFMM has very little `accessory` data: 455 train images with it, and only 57 in val. Dropping `accessory` from the mean changes the run comparison from −0.0096 to −0.0008.
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

## What is still unknown
- Which raw class (glove, scarf, sunglasses) fails on LIP and CIHP. This needs per-raw-class evaluation (not done).
- Why eyeglass pixels go to `face`. Not tested.
- Whether DFMM neckwear and tie masks are as clean as eyeglass. Not viewed.

## Next
Per-raw-class evaluation needs inference on the saved checkpoint (GPU or slow CPU), so it waits for a go-ahead.
