# M10 Data concerns from model results (X2–X4)

Task M10, 2026-11-07. Handoff: `docs/handoffs/2026-11-07-M10-lemasc.md` (method, commands, gotchas).

**In short:** some weak classes come from the data, not the models.

- LIP jumpsuits and CIHP socks are label conflicts in the merged label set.
- Skirt is mostly confused with dress, and the cause is mixed.
- Outer → upper is mostly a model error.

No retrain is needed. What T9 and T10 should do is listed under [Decisions](#decisions).

## Runs and sources

| Alias | Run | Model |
|---|---|---|
| X2 | `20261105-1307-1acb88f-unet-r34` | U-Net ResNet-34 |
| X3 | `20261105-2050-f4e2ffe-deeplabv3p-r50` | DeepLabV3+ ResNet-50 |
| X4 | `20261105-2220-ac9a26b-segformer-b2` | SegFormer-B2 |

- **IoU:** from `runs/<run>/eval/eval-v2/<dataset>/metrics.json`.
- **Raw-class and per-image counts:** from `scripts/raw_class_breakdown.py`, in `data/review/M10/breakdown/`. These counts sum to the saved metrics: X2 and X3 exactly, X4 within GPU numeric difference (handoff, Gotchas).
- **Visual review:** `notebooks/review/m10_data_concerns.py`, config `configs/review/m10.yaml`. The verdicts and decisions are in `docs/reports/2026-11-07-M10-verdicts.csv`.

Values are X2 / X3 / X4 unless a table says otherwise. "Recall" of a raw class is the share of its pixels predicted as the shared class it maps to. A raw class is a class of the original dataset, before the merge into the 12 shared classes.

## Questions

| # | Question | Why it was raised |
|---|---|---|
| Q1 | Is LIP `dress` weak because LIP maps jumpsuits to dress, while CIHP labels them as upper + pants? | LIP `dress` IoU 0.30 / 0.34 / 0.40, far below CIHP and DFMM. |
| Q2 | Is CIHP `footwear` weak because feet are small, and do socks pull it down? | CIHP `footwear` IoU 0.40 / 0.42 / 0.50. Socks are 23% of CIHP footwear pixels. |
| Q3 | Is skirt ↔ pants confusion partly shorts? LIP and CIHP have no shorts class. | `skirt` IoU 0.24–0.34 on LIP and CIHP; 17–28% of skirt pixels are predicted as pants. |
| Q4 | Is upper ↔ outer confusion label ambiguity? | 10–18% of pixels go each way on LIP and CIHP, in all models. |
| Q5 | Are X3's lower DFMM scores on fine classes real? | X3 − X2 on DFMM is −0.011 to −0.023 on hair, skin, footwear and headwear. |
| Q6 | Can DFMM `accessory` IoU support any claim? | 0.0007 / 0.41 / 0.61. |

## How the visual review was done

- **Images:** 24 test images per question: 12 "worst" (lowest X4 recall of the raw class) and 12 random (seed 0). Each image covers at least `min_share` of the class.
- **Who reviewed:** the maintainer reviewed every image of Q2, Q3 and Q4 (72 images).
- **Q1:** not reviewed; it is answered from the numbers.
- **Which half answers what:** only the random half shows how common a cause is. The worst half shows what the hardest failures look like.

"Where the pixels go" tables show the mean, over the images of a verdict, of the share of the raw-class pixels that each run predicts as each shared class. They are in the notebook, section 4.

## Findings

### Q1: yes, in part. LIP jumpsuits pull LIP `dress` down.

| Raw class (test) | Maps to | Recall | Most of the rest goes to |
|---|---|---|---|
| LIP dress | dress | 0.49 / 0.55 / 0.60 | upper 0.16–0.21 |
| LIP jumpsuits | dress | 0.24 / 0.39 / 0.42 | pants 0.19–0.26, upper 0.19–0.24 |
| CIHP dress | dress | 0.64 / 0.68 / 0.70 | upper 0.12–0.14 |
| DFMM rompers | dress | 0.93 / 0.95 / 0.95 | |

- **The training data teaches two answers.** In the train split, jumpsuits are 27.3% of LIP `dress` pixels (390 images). CIHP has no jumpsuit class and labels them as upper + pants. The models often answer the CIHP way.
- **One-piece garments are not hard in general.** DFMM rompers are found at 0.93–0.95.
- **Jumpsuits explain only part of the gap.** Real LIP dresses are also below CIHP dresses.

### Q2: CIHP shoes are small; CIHP "socks" are often tights, long socks or bare-looking legs.

**Shoes:**

- Shoe recall rises with size in every dataset, and CIHP shoes are small: median 0.89% of the image, against 2.46% on LIP.
- At the same size, CIHP stays below LIP by 0.05–0.18 for X2 and 0.01–0.08 for X4. The size is measured per image, and CIHP images often show several people, so part of that gap may also be size.

**Socks:**

- CIHP socks recall is 0.25 / 0.32 / 0.45, against 0.71 / 0.69 / 0.77 on LIP.
- **Proxy check:** an image counts as tights-like if it has a skirt or dress and little visible leg skin. Tights-like images make up 41% of CIHP socks pixels, against 6% on LIP.

Review, 24 of 24:

| Verdict | Random | Worst | Where the pixels go (X4) |
|---|---|---|---|
| K: long socks, tights or stockings | 6 | 10 | pants 0.50, skin 0.19, footwear 0.17 |
| L: wrong label (looks like bare skin) | 4 | 2 | skin 0.58, footwear 0.23, pants 0.16 |
| S: short socks, clearly visible | 2 | 0 | footwear 0.69, skin 0.28 |
| M: clear socks, model mistake | 0 | 0 | |

- **What CIHP labels as socks:** in 10 of 12 random images, it covers something that looks like trousers (K) or bare legs (L). The models predict exactly that: pants or skin. No image was a clear model mistake.
- **Where the review and the proxy agree, and where they don't:**
  - The proxy flags all 4 random "L" images as tights-like. These are skin-coloured tights or bare legs labelled socks. Neither can be told apart from the image, and both have the same effect on the scores.
  - The proxy undercounts long socks: 10 of the 16 "K" images have no skirt or dress, for example leggings under shorts.
- **Clearly visible socks are found.** The 2 "S" images reach X4 recall 0.99 and 0.38.

### Q3: skirt is lost mostly to `dress`; pants come second, with mixed causes.

Review, 24 of 24:

| Verdict | Random | Worst | Where the pixels go (X4) |
|---|---|---|---|
| D: skirt vs dress unclear (looks one-piece) | 1 | 5 | dress 0.83, outer 0.16 |
| Y: short skirt that looks like shorts | 6 | 0 | skirt 0.54, pants 0.24, outer 0.13 |
| M: clear skirt, model mistake | 3 | 6 | dress 0.42, skirt 0.31, pants 0.11 |
| L: wrong label (shorts or trousers) | 2 | 1 | pants 0.61, upper 0.18 |

- **The worst images are lost to dress.** On the worst half, all runs predict 0.68–0.73 of the skirt pixels as dress and only 0.13–0.16 as pants. 5 of the 6 LIP worst images look like one-piece outfits labelled as skirt.
- **No single cause dominates the random half.** Of 12 random images: 6 shorts-like (Y), 3 model mistakes (M), 2 wrong labels (L), 1 skirt-or-dress (D).
- **Where the pants confusion comes from:** shorts-like skirts and wrong labels. Clear skirts mostly go to dress.
- **How the original question changes:** "skirt vs pants" is better described as "skirt vs dress, then shorts".
- **Training data:** LIP has few skirts (535 train images, against 1,972 in CIHP). That may add to the low LIP score; the review cannot test it.

### Q4: outer → upper is mostly a model error.

Review, 24 of 24:

| Verdict | Random | Worst | Where the pixels go (X2 / X3 / X4) |
|---|---|---|---|
| O: clear outer layer, model mistake | 11 | 6 | outer 0.52 / 0.51 / 0.38, upper 0.39 / 0.37 / 0.51 |
| A: ambiguous garment | 1 | 4 | upper 0.73 / 0.65 / 0.59, background ~0.25 |
| U: not an outer layer | 0 | 2 | upper 0.89 / 0.96 / 0.94 |

- **The coat labels are mostly right.** 11 of 12 random images are clear outer layers. `upper` and `outer` each come from a single raw class in every dataset, so our mapping did not create this confusion.
- **Ambiguous garments are rarer and concentrate in the worst half.** All runs call them upper.
- **On the random half, all three runs are similar:** outer recall 0.66 / 0.59 / 0.60. X4 looks weaker on the "O" images only because the worst images were picked by X4 recall.

### Q5: the X3 DFMM drops are larger than run-to-run noise.

- **How noise was measured:** X1 and X2 have the same config. Except for `accessory`, their DFMM IoU differs by at most 0.0078 per class.
- **X3's drops exceed it:** X3 − X2 is hair −0.012, skin −0.013, footwear −0.011 and headwear −0.023.
- **Caveat:** this noise comes from one pair of U-Net runs. DeepLabV3+ may vary more between runs.

### Q6: no.

- **Too few examples:** DFMM `accessory` has 72 test images and 157,182 test pixels (0.015% of labelled pixels).
- **Unstable:** between the two U-Net runs alone, its IoU changes by −0.207.

## Decisions

The maintainer decided these on 2026-11-07, after the review. All are recorded in the verdicts file as `d1`–`d4`. None of them changes a frozen contract or needs a retrain.

| # | Finding | Decision | For |
|---|---|---|---|
| d1 | LIP jumpsuits mapped to `dress` | Limitation, and also report LIP `dress` without jumpsuit pixels. Q1 images were not reviewed; decided from the numbers. | T9 (extra score), T10 (limitation) |
| d2 | CIHP socks (merged into `footwear`) are often tights, long socks or bare-looking legs | Limitation, and also report CIHP `footwear` without socks pixels | T9 (extra score), T10 (limitation) |
| d3 | skirt lost to dress, and to pants for shorts-like skirts | Limitation for skirt vs dress and for shorts-like skirts; error analysis treats clear skirts (M) as model error | T10 |
| d4 | outer predicted as upper | Model error in the error analysis | T10 |

The decision options in `configs/review/m10.yaml` say "T9 error analysis". In `PLAN.md`, error analysis is T10, so this table names T10.

## Limits of this check

- **Sample size:** each review rate rests on 12 random images per question. These rates show what is common; they are not precise percentages.
- **Proxies:** the tights proxy and the shoe-size groups are measured per image, but CIHP images often show several people.
- **One reviewer:** all verdicts come from the maintainer alone. "L" verdicts marked "skin?" and a few others were noted as uncertain.
- **X4 recomputed elsewhere:** X4 was evaluated again on another GPU (RTX 4070 instead of RTX 3060). At most 0.00044 IoU per class differs from its saved metrics.

## Follow-ups (outside M10)

- **Scores without some raw-class pixels (d1, d2):** T9 needs an eval option that leaves out the jumpsuit or socks pixels. The breakdown here gives recall per raw class, not IoU.
- **CIHP `skin` includes `torso_skin` (26.8% of its train pixels); LIP has no such class.** How LIP labels exposed torso skin is not checked.
- **Occlusion:** the reviewer noted that a person in front can confuse the model (`cihp:3050`) and that depth might help.
