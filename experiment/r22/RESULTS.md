# r22 — results: 16x reaches 100% immediately; the vocabulary run was confounded (see r23)

## Run 1 — the 16x store (box1, M=8192, 64 segments, 8,128 slots/example, batch 2 x accum 16, 500 steps,
## `--init experiment/r20/ckpt/r20_warm_skipnorm_M4096.pt`)

| step | 125 | 250 | 375 | 500 |
|---|---|---|---|---|
| recall | **100%** | **100%** | **100%** | **100.0%** |
| slot@1 | 100% | 100% | 100% | 100% |
| read mass | 0.96 | 0.95 | 0.95 | 0.95 |

Content-miss and match-miss 0% at every eval; alignment 100% at offset 0; val ppl 2.00.

**H22a (16x reaches >= 95% warm-started) — confirmed**, and at the *first* eval. The curriculum's opening number now
reads: 4x 79.7% -> 8x 96.2% -> 16x 100%. Each doubling transfers more easily than the one before, and read mass has
barely moved across a 16-fold range of store sizes (0.97 -> 0.95).

Its matched scrambled control runs in r23 (box1) — this round shipped the real run without one, which r23 fixes.

## Run 2 — key vocabulary (box2, 1x store, cold, 4000 steps)

| run | symbols | facts | chance | final | gate | read mass |
|---|---|---|---|---|---|---|
| r22_1x_sym128_f48 | 128 | 48 | 0.8% | **0.8%** | 0.003 (shut) | 1.00 -> 0.80 (decaying) |
| r22_1x_sym32_f8 (baseline) | 32 | 8 | 3.1% | **56.1%**, still climbing | opening | 0.99 |

**H22b — not answered. The run was confounded and the result must not be read as a vocabulary finding.** Two variables
moved at once: symbols 32 -> 128 *and* facts 8 -> 48. The failure is real (flat at chance for 4000 steps, the gate
closing and mass decaying — the r16/r17 signature of a run that never starts using the store), and it is not a box or
setup artefact, since the matched baseline lifted off on the same box, seed and step budget. But it cannot be
attributed to either variable.

A second gap: the baseline ended at 56.1% *still rising* because it got 4000 steps where r18's identical configuration
got 8000 (and reached 100%). So "did not lift off" was also confounded with "was not given the budget".

r23 fixes both — 8000 steps, and the vocabulary change tested with the fact count held at 8.

## Diagnosis

The store-size axis looks finished: 16x works immediately, mass is flat, and nothing suggests a capacity wall — what
ends the axis is compute per step (16x needs batch 2 and 4x r20's time), not recall.

The interesting failure is on the other axis, and this round's design was too coarse to name it. Writing it up as a
confound rather than as a vocabulary limit is the honest call; r23 and r24 do the one-variable versions.

## Next

- r23 (running): 128 symbols with **8** facts (vocabulary alone), the 32/8 reference at 8000 steps, 128/48 at the full
  budget, and the 16x scrambled control on box1.
- r24 (running): the fact count swept at a fixed 128-symbol vocabulary (16, 24, 32 facts) to find where 1x breaks.
