# r16 — results: the plain skip path does not lift off at the 4x store

## What ran

| run | value | store | steps run | recall at stop | gate | read mass | correct-slot reads |
|---|---|---|---|---|---|---|---|
| r16_skip_M2048 | last block + block-0 skip (sum) | 4x, 1,920 slots, TinyStories | 5000 of 12000 | **2.8%** (chance) | 0.010 | 0.54 | 100% |
| r16_obs_M2048_shuf | — | — | not run | — | — | — | — |

Stopped by the rule set in `STATE.md` before the round: still at chance at step ~5000 with the gate below 0.03.

```
recall  (every 250 steps)   3 1 3 3 2 3 2 2 4 3 3 3 3 3 5 3 3 2 1 3
gate                        .002 .004 .009 .016 .020 .020 .017 .014 .013 .012 .011 .010 .010 .011 .010 .010 .010 .011 .011 .010
read mass                   1.00 1.00 1.00 .97 .92 .87 .83 .79 .75 .72 .69 .66 .64 .62 .60 .59 .58 .56 .55 .54
```

## Verdict

- **H16a (the objective placement holds at 4x) — not met** in 5000 steps, with the stop rule applied rather than the full
  12000. The block-0 store at the same size was at 100% by step 5500 (r15); this run shows no sign of starting.
- **H16b (4x scrambled control)** — not run; moved to r17.

## Diagnosis

Three signals separate this from a slow lift-off:

1. **The gate closes instead of opening.** It peaked at 0.020 (step ~1500) and fell to 0.010, where every successful
   4x run had a gate climbing past 0.1 by step 2000. The read path is being taught that injecting what it reads hurts.
2. **Read mass keeps falling** — 0.54 at step 5000 and still going, where the block-0 store's pre-lift-off dip bottomed
   at ~0.81-0.84 and recovered.
3. **Matching stays at 100%**, so the right slot is found; what it holds is not usable.

The stored value is last-block state + block-0 output. At 512 slots (r11) that sum worked. At the 4x store, with many
more distractor positions reading the store and the last-block representation now trained on a non-memorised corpus,
the sum apparently no longer lets the read path recover the block-0 half. The obvious candidate is scale: if the
last-block state's norm dominates, the identity-bearing half is a small perturbation. This is a hypothesis, not a
measurement (norms were not logged); r17 tests the fix directly rather than chasing the mechanism.

## Next

r17: `--value-skip-norm` stores `mean|skip| * (unit(last) + unit(skip))`, so both halves carry equal weight. Plus the
4x scrambled control carried over from here.

## Caveat added during r17 — the stop may have been early

Lift-off times already measured suggest the skip path at 4x might not start until ~11000 steps: the block-0 store
lifted off ~5x later at 4x than at 512 slots (~750 -> ~3750), and the skip path at 512 slots lifted off at ~2250.
Stopping at step 5000 was therefore within the plausible pre-lift-off window. The closing gate (0.020 -> 0.010) is
still a real difference from every run that later worked, but "stayed at chance for 5000 steps" alone is not evidence
the configuration cannot work at 4x. r17 runs its full budget unless its gate collapses.
