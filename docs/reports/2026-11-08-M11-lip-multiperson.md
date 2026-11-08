# M11 — Other people in LIP images

## In plain words
LIP labels **one person per image**. Anyone else in the photo counts as background. CIHP, which we also train on, labels **every person**. So the models also parse the extra people in LIP test images, and each of those pixels is scored as wrong.
In 45 LIP test images checked by eye, this was the cause in 28. The check covered the images with the most of these errors, plus a random draw.
Ignoring these large areas would raise LIP mIoU by about 0.04–0.05 for every model, but the ranking of the models and the gaps between them stay the same. This is a difference in labelling rules between the datasets, so no change to the model can fix it. We record it as a limitation.

## Why
T10 found that 58% of X4's wrong `background` pixels on LIP are **interior**, meaning not at an edge. T10 suggested a cause (other people who are not labelled) but did not check it. M11 checks it.

## Data and method

- **Runs:** X2 `20261105-1307-1acb88f-unet-r34`, X3 `20261105-2050-f4e2ffe-deeplabv3p-r50`, X4 `20261105-2220-ac9a26b-segformer-b2` (`splits-v2-d023252b81d9`, `eval-v2`, `best.pt` step 40000). Test sets: LIP for all three runs, CIHP for X4 as the contrast.
- **Prediction:** `scripts/multiperson.py configs/report/m11.yaml --wait` predicts again on the RTX 4070, using the same steps as `evaluate.py`, which is unchanged. It held the GPU lock for about 6 minutes.
  - X2 and X3 produce the same confusion matrix as their `metrics.json` (max diff 0).
  - X4 differs by up to 2,359 pixels (LIP) and 2,898 pixels (CIHP) in a single cell, because its `metrics.json` came from the RTX 3060. T10 saw the same thing.
  - Outputs are in `data/analysis/M11/` and `data/review/M11/`.
- **Tables:** `scripts/m11_results.py configs/report/m11.yaml [--sheets]` uses no GPU. Every number in this report comes from [`2026-11-08-M11-results.json`](2026-11-08-M11-results.json). [`2026-11-08-M11-tables.md`](2026-11-08-M11-tables.md) has the full tables.
- **Definitions** (`src/cloth_segment/multiperson.py`):
  - **wrong-person pixel:** ground truth is `background` and the prediction is any other class.
  - **interior:** outside the T10 edge band (1% of the image diagonal).

## 1. How big and how concentrated

| run | test set | interior wrong-person px / GT background px | images with any |
|---|---|---|---|
| X2 | LIP | 0.0707 | 9,200 / 9,632 |
| X3 | LIP | 0.0657 | 9,135 / 9,632 |
| X4 | LIP | 0.0614 | 9,125 / 9,632 |
| X4 | CIHP | 0.0146 | 3,908 / 5,000 |

- **The rate on LIP is about 4 times the rate on CIHP.** CIHP labels every person, so an extra person there is a correct prediction.
- **A few images hold most of it.** For X4 on LIP, the top 10% of images hold 71% of these pixels, and the top 25% hold 91%. So the errors come as large blobs in a small number of images, not as small noise everywhere.
- **Small people are not the main reason.** The median person area in the top 10% of images is 0.35 of the image, against 0.44 for all LIP test images. That is lower, but these are not tiny-person crops.
- **The strict test finds only a small part.** The strict test counts a predicted person that does not touch the labelled person at all. It finds only 12–14% of the pixels on LIP, because other people usually touch or overlap the labelled one in the photo. This is why the visual check below was needed.

## 2. Visual check

- **What was checked:** X4 on LIP, 45 images: the 30 images with the most interior wrong-person pixels, plus 15 drawn at random (seed 0) from the other images with at least 200 such pixels.
- **What each sheet shows:** image | GT | prediction | wrong pixels in magenta, drawn to `data/review/M11/sheets/`. The sheets are dataset images, so they are not committed.
- **Who read them:**
  - A Claude subagent read all 45.
  - A second Claude session checked 18 of them (sheets 0, 2 and 6) and changed one verdict (#40: the GT labels both women, and the wrong pixels are small gaps inside the people already labelled).
  - A maintainer scanned all 8 sheets and agreed with the pattern. They changed two verdicts on sheet 4 (#26 and #29: the GT leaves the main person unlabelled) from `other_person` to `main_annotation_missing`.
  - [`2026-11-08-M11-verdicts.csv`](2026-11-08-M11-verdicts.csv) keeps the first reading, which images were checked, and the final verdict.

| verdict | top 30 | random 15 | total |
|---|---|---|---|
| other person(s) parsed | 24 | 4 | 28 |
| main person not (fully) annotated in GT | 6 | 0 | 6 |
| object or clutter (bag, bat, toy) | 0 | 4 | 4 |
| mannequin, poster or reflection | 0 | 0 | 0 |
| unclear (small areas) | 0 | 7 | 7 |

- **The large blobs are other people.** This held in 24 of the top 30 images. In some of them, the person LIP labelled is not the main person in the photo, for example someone at the edge while the subject in the middle is background.
- **The other 6 of the top 30 come from the same rule, seen from the other side.** The clear subject of the photo is not labelled in the GT at all.
- **The random group is mostly small, mixed errors.** These 15 images hold only 0.2% of all interior wrong-person pixels on the test set; the top 30 hold 9.1%.

## 3. Effect on the scores (analysis only)

**Rule:** take each interior wrong-person blob (8-connected) that covers at least a set share of the image, plus the wrong-person pixels within the edge band width of it, and leave those pixels out of the score.

**Limit:** this rule cannot tell another person from an object or an unlabelled main person, so the change is an **upper bound**. **It is not an official score.**

mIoU over 11 classes, without `accessory`:

| run | test set | official | blob ≥ 0.5% | blob ≥ 1% | blob ≥ 2% | images affected (1%) |
|---|---|---|---|---|---|---|
| X2 | LIP | 0.5924 | 0.6428 | 0.6386 | 0.6330 | 3,807 |
| X3 | LIP | 0.6071 | 0.6549 | 0.6506 | 0.6447 | 3,682 |
| X4 | LIP | 0.6399 | 0.6879 | 0.6838 | 0.6783 | 3,311 |
| X4 | CIHP | 0.6682 | 0.6802 | 0.6781 | 0.6756 | 708 |

- **LIP gains +0.043 to +0.046 at 1% for every model; CIHP gains only +0.010.**
- **The ranking and the gaps do not change.** X4 − X2 is +0.047 in the official score and +0.045 at 1%; X4 − X3 is +0.033 in both. **This problem does not affect the model comparison.**
- **Which classes it touches:** for X4 on LIP at 1%, the classes that rise most are `background` (+0.051), `skin` (+0.067), `hair` (+0.056), `pants` (+0.052) and `face` (+0.051). **`dress` (+0.021) and `skirt` (+0.016) barely move**, so this problem is not why those two classes are low (T9, T10).

## 4. Conclusion

- **The T10 hypothesis holds.** Most interior background errors on LIP come from people that LIP does not label.
- **The cause is the fusion of the datasets.** LIP and CIHP use different rules for who gets labelled: LIP labels one person, CIHP labels everyone. The same prediction counts as right on CIHP and wrong on LIP. This is evidence for the label conflicts between datasets that we already list in the limitations.
- **We will not fix it.** We would have to change the LIP test set (rule D4, and "Never alter the protocol or test sets to improve a number" in CONTRIBUTING section 4), or train the model to label only one person, which would break the CIHP rule. **We record it as a limitation.**

## Limitations of this analysis

- The verdicts come from AI readers (one reading plus a check of 18 images) and one maintainer's scan of the sheets, and come from 45 images only. The random group is too small to give rates.
- The score change is an upper bound (section 3).
- All analysis predictions ran on the RTX 4070. X4's official scores came from the RTX 3060; the difference is shown above.

## Next

- **T11:** use this as a limitation and dataset-fusion slide: one or two example images, plus the table in section 3. Check the LIP licence before showing images (T9 open item).
