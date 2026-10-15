# r21 — is the 8x result seed-luck, and does it hold more facts?

## Ideate (from r20)

The 8x store (M=4096, 32 segments, 4,064 slots per example) warm-started from the 4x model reaches 100% by step 500 —
faster than the 4x warm start did (r19: 79.7% at step 250 vs 96.2% here). Every eval so far reads the correct slot with
mass 0.97. Two things are untested at 8x, and both are cheap:

1. **Seed.** Every curriculum step so far (2x -> 4x -> 8x) ran at seed 0. r4 is the cautionary case: an 8-fact result
   that held on one seed and collapsed on two others.
2. **Facts per store.** 8 facts over 4,064 slots is a sparse store. r11 got 32 facts at 1x, but not since the store grew.

Growing to 16x is deferred: at n_seg 64 / M 8192 one example needs batch 1 and ~4x the compute per step of r20, so it
does not fit the ~1 h/box budget. Robustness at 8x is the better use of this round.

## Hypotheses

- **H21a — seed 1 reaches >= 95% at 8x** from the same 4x checkpoint. Falsified if it stalls near chance: the curriculum
  would then be carrying seed-specific structure, and the 2x -> 4x step needs rerunning per seed.
- **H21b — 32 facts reach >= 95% at 8x.** Falsified if recall drops while slot@1 stays ~100%: that is a content failure
  (the r6 signature), meaning the store holds the binding but the value cannot carry 32 distinct facts at this size.

## Falsifiers, watched at every eval

- slot@1 below ~100% -> a matching failure, not content; report which (miss taxonomy).
- Read mass falling below ~0.9 -> the read is diluting at 8x and the numbers above were seed luck.

## Runs (as r20: M=4096, 32 segments, batch 4 x accum 8, 1500 steps, --init r19_warm_skipnorm_M2048.pt)

| box | name | change |
|---|---|---|
| box1 | r21_8x_seed1 | `--seed 1` |
| box2 | r21_8x_f32 | `--n-facts 32` |

Also on box1 once r20's checkpoint exists: two-process persistence at 8x (`mech.persist`, `--batch 2`, store files are
~4x r19's 33 MB).
