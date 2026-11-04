# Handoff — D4 empty-mask rule for test sets (chore, no task ID)

**Status:** PR #3, waiting for review.

## In plain words
D4 said "the test sets are not changed", but `splits-v1` already drops 28
background-only images from the LIP test set. We checked all 28 by eye: each
shows a person in visible clothing, so the empty mask is an annotation
failure. D4 now says the exclusion rules apply to train and test alike.

## Summary
- `splits-v1` (T3, rule 6) excludes every image whose mask has only
  background. For LIP val (our test set) that is 28 of 10,000 images, so the
  test set has 9,972.
- D4 and the data disagreed. This PR rewords D4 to match the data. No code,
  data or split change. Data version stays `splits-v1-47206b9adcd9`.
- The wording has no counts and no thresholds. Those live in the splits
  version, so later filters (see Open items) do not need another D4 edit.

## Decisions and why
- **Same rule for train and test.** A person in the image with an all-background
  mask would count as false positives for any model that finds the person.
  That punishes a correct prediction.
- **Consequence:** our LIP numbers are not comparable to published LIP val
  numbers (9,972 images, not 10,000). Say so in the report (T9/T10).

## Results / findings
Evidence for "annotation failure", checked on all 28 images (LIP val, source
`TrainVal_images/val_images/<id>.jpg`):

- All 28 show at least one clearly visible person wearing clothes.
- Most are crowd or sports crops (tennis, baseball, spectators) where the
  labelled person is partly hidden or small. Two or three are very blurry or
  dark. None is a picture without a person.
- Source of the 28 vs the rest: 28 val + 56 train empty-mask exclusions = 84
  (the T1 scan found 105 empty train/val masks; the other 21 train images were
  already excluded as leaks, which has priority).
- CIHP has no empty masks, so no CIHP images are affected.

IDs: 100985_570554, 122962_1210087, 127306_1242928, 154004_216643,
166776_1215521, 16693_1301135, 201736_1188651, 241305_1733806,
294223_900100294223, 310518_1208237, 314949_1699710, 316971_1286020,
333984_1218078, 335871_1302956, 360279_504351, 382142_1296572,
460458_519262, 483769_1227277, 495266_1301603, 497273_1214430,
506315_1253081, 53712_1723089, 538113_1240174, 541092_1240809,
54130_1319713, 572145_1221914, 579901_1726813, 98656_262405

## Gotchas
- "Empty" is computed on the mapped label set (c1..c11 sum is zero), not on the
  raw LIP mask. For val the two give the same 28 images.

## Open items
- A second filter for very small and very thin LIP images (short side under
  64 px, ratio worse than 6:1) is proposed in a follow-up PR with
  `splits-v2`. This PR does not include it.
- Report metrics by image size and shape in T9/T10.

## Next
Follow-up PR: size and shape filter (`splits-v2`). Both must merge before the
splits contract is frozen (when T4 merges).
