# r23 — results: the key vocabulary is not the limit; the fact count is

## Numbers (1x store, cold, 8000 steps, box2; r18's configuration — plain block-0 skip, batch 32, seed 0)

| run | symbols | facts | chance | 1000 | 2000 | 4000 | 6000 | final (8000) | read mass | slot@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| **r23_1x_sym128_f8** | 128 | 8 | 0.8% | 0.8% | 1.4% | 11.7% | 81.9% | **99.7%** | 1.00 | 100% |
| r23_1x_sym32_f8 (reference) | 32 | 8 | 3.1% | 5.0% | 84.4% | 100% | 100% | **100.0%** | 0.99 | 100% |
| r23_1x_sym128_f48 | 128 | 48 | 0.8% | 0.8% | 1.6% | 0.8% | 0.3% | **0.9%** | 0.97 -> **0.70** | 100% |

## Verdicts

- **H23a (128 symbols with 8 facts reaches >= 95%) — confirmed.** 99.7%, against a 0.8% chance line. **The key
  vocabulary is not the limit.** It costs *time*, not recall: lift-off moves from ~step 1500 (32 symbols) to ~step 4500
  (128 symbols), roughly 3x, and the run was still improving at the end.
- **H23b (32/8 reproduces r18 on box2) — confirmed.** 100% by step 4000, matching r18's 1x result on the other box. The
  two boxes are comparable (torch 2.14 vs 2.13 makes no difference here).
- **H23c (128/48 stays below 50% even at 8000 steps) — confirmed.** 0.9% at the end, i.e. exactly chance, with twice
  r22's budget. It is not slow; it never starts.

Together these attribute r22's failure cleanly: **the fact count, not the vocabulary.**

## Diagnosis

Read mass is the discriminator, and it splits the three runs perfectly:

| run | mass over training |
|---|---|
| 128/8 | 0.98 -> 1.00 (rises into lift-off) |
| 32/8 | 0.99 -> 1.00 |
| 128/48 | 0.97 -> **0.70** (decays monotonically) |

Every run finds the correct slot 100% of the time. The two that succeed *concentrate* read weight on it; the failure
spreads it out. This is the r16/r17 signature again: without early successful reads there is no pressure keeping key
representations separated, the read dilutes, and the gate closes rather than opens.

What is different about 48 facts is not the store — 508 slots hold 48 facts easily, and 32 facts fit a 4,064-slot store
at 100% (r21). It is the **query segment**: with `n_facts` queries in one segment, 48 facts means 48 queries competing
in a single 128-token segment, each needing its own read. That is a plausible mechanism and not yet a measurement;
r24 sweeps 16/24/32 facts at 128 symbols to find where it breaks, which also distinguishes "gradual crowding" from
"a cliff".

The vocabulary result has a practical reading too: 128 distinct keys work, so the earlier cap of "32 facts" was an
artefact of the 32-symbol vocabulary rather than anything about the memory.

## The 16x scrambled control (box1)

r22's 16x run reached 100% at every eval and shipped without a control. `r23_16x_shuf` is the matched scrambled
version — same warm start, same 500 steps, only `--shuffle-store` differs:

| step | 125 | 250 | 375 | 500 |
|---|---|---|---|---|
| recall | 0.0% | 2.5% | 0.0% | 7.5% |

**H23d — confirmed, with the eval size stated.** Against 100% for the real store. Chance is 3.1%, and the eval here is
only n=40 (batch 2 x 20 iterations, because a 16x store needs batch 2), so one correct answer is worth 2.5 points:
the final 7.5% is 3 hits where chance expects 1.25, which is ordinary Poisson noise, not a leak. The mean across the
four evals is 2.5%.
