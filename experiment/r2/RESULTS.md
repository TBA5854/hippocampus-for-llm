# r2 — results: early keys fix content, not matching; r1's diagnosis was half wrong

## Numbers (2 facts, 4000 steps, 5 runs in parallel)

| run | write | key from | shift | best | final | identity-slot reads (final) |
|---|---|---|---|---|---|---|
| r2_w4 | 4 | write | 1 | 100.0% | **100.0%** | 0.0% |
| r2_w5s1 | 5 | write | 1 | 71.6% | 10.6% | 0.0% |
| r2_w5s0 | 5 | write | 0 | 79.8% | 79.4% | 0.0% |
| r2_w5s0k1 | 5 | layer 1 | 0 | 58.3% | 58.1% | 59.4% |
| r2_w5s0k1_shuf | 5 | layer 1 | 0 | 4.2% | 4.2% | — (control) |

```
w4      3  2  3 52 77 92 99 96 99 99 100 100 100 100 100 100
w5s1    4 44 59 65 72 62 51 33 16  9   8  15  10  11  10  11     <- r1's collapse, replicated
w5s0    4  3  3  8  9  6 12 16 19 34  47  61  73  75  80  79     <- late climber
w5s0k1  3 46 53 49 50 52 57 57 55 55  54  54  53  55  58  58     <- pinned near 1/2
```

## Verdicts

- **H2a (the collapse is content) — falsified.** `w5s1` replicated r1's collapse (71.6% -> 10.6%), and
  r3's rerun with the offset diagnostic shows its alignment scattering as it falls (see r3). The
  collapse is a *matching* breakdown.
- **H2b (shift 0 fixes the value) — not falsified, contrary to the mid-round read.** `w5s0` looked dead
  at step 750 (2.8%) and climbed late to 79.4%. Judging a run at a quarter of its budget was wrong, and
  I said so at the time rather than acting on it. It pairs on something other than identity slots.
- **H2c (early keys) — partly.** Content transfer is nearly perfect (content-miss ~11% at the end, 3% at
  step 750), but recall pinned near 1/2: the read finds *a* key position without telling the two keys
  apart. Cause, fixed in r3: stored keys are raw layer-1 states, but the query passed through a randomly
  initialised projection, and the key side cannot learn (it enters the store detached).

## Correction to r1

r1 concluded "block 4 keeps token identity, block 5 loses it". **`r2_w4` — the 100% reference — made
0% of its reads on identity slots**, after 43-48% early in training. Block 4 also pairs by prediction:
the slot one *before* `k_j`, whose state predicts `k_j`, which a tied head puts in token space where a
layer-1 query for `k_j` can match it. r3's `w4_align` confirms it reads at offset +1 in 100% of cases.
So identity-versus-prediction does not explain why block 5 collapses and block 4 does not. That
mechanism is left open deliberately (mech-interp is deferred per the project's standing notes); the
same-space-key placement below does not depend on it.

## Also measured

- 5 parallel runs are ~15% slower per run than sequential (GPU compute-bound); each holds 2.77 GB, and a
  6th OOMs. Rounds run sequentially from r3.
