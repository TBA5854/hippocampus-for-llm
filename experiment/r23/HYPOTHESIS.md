# r23 — separate key vocabulary from fact count (r22's confound), and control the 16x store

## Ideate (from r22)

r22's box2 runs changed **two** things at once: symbols 32 -> 128 *and* facts 8 -> 48. The 128/48 run stayed at chance
for 4000 steps with the gate closing (0.003) and read mass decaying (1.00 -> 0.80); the matched 32/8 baseline lifted off
on the same box and seed (10.6% at 2000, 29.8% at 3000, 56.1% at 4000, mass 0.99, gate opening). So the failure is real
and not a box artefact — but it cannot be attributed to the vocabulary, because the fact count moved too.

Also: the 32/8 baseline ran 4000 steps and ended at 56.1% *still climbing*, where r18's identical 1x configuration ran
8000 steps and reached 100%. Every r23 run therefore gets 8000 steps, so "did not lift off" is not confounded with
"was not given r18's budget".

## Hypotheses

- **H23a — 128 symbols with 8 facts reaches >= 95%** at 1x in 8000 steps. This is the vocabulary change *alone*.
  Confirmed -> the key vocabulary is not the limit and r22's failure was the fact count. Falsified (flat at chance,
  gate closing) -> a 128-symbol vocabulary is itself the blocker, and the diagnostic is whether read mass decays as it
  did in r22.
- **H23b — 32 symbols with 8 facts reaches >= 95%** in 8000 steps, reproducing r18 on box2. This is the reference line;
  if it fails, box2 differs from box1 in some way that invalidates comparing across boxes (torch 2.14 vs 2.13 is the
  known difference).
- **H23c — 128 symbols with 48 facts stays below 50%** even at 8000 steps, i.e. r22's failure was not merely slowness.

Together a/b/c attribute r22's negative result to exactly one of: vocabulary size, fact count, or budget.

- **H23d — the 16x store's scrambled control is at chance.** r22's 16x run has no control yet; every store size so far
  has had one at matched settings.

## Runs

| box | name | change (all at the 1x store, 8000 steps, cold) |
|---|---|---|
| box2 | r23_1x_sym128_f8 | `--n-symbols 128 --n-facts 8` — vocabulary alone |
| box2 | r23_1x_sym32_f8 | reference line (r18's configuration, box2) |
| box2 | r23_1x_sym128_f48 | r22's failing run, with the full budget |
| box1 | r23_16x_shuf | 16x store, `--shuffle-store`, matched to r22_16x (warm-started, 500 steps) |

Note for later: 48 facts is only reachable *with* a bigger vocabulary (32 symbols cap facts at 32), so "facts" and
"vocabulary" cannot be fully crossed — the 32-symbol/48-fact cell does not exist. The comparison that does exist is
128/8 vs 128/48, and that is what H23a + H23c measure.
