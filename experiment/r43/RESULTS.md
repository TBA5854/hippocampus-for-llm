# r43 — the curriculum was undertrained: 2x steps takes 8 facts from 46% to 74%

r38/r40 warm-started 8 facts and got 46% at 6000 steps, with the gate still climbing. This asks whether the gap to
the 90.3% four-fact ceiling is a real limit or just training budget. 12000 steps, warm from the r38/r40 checkpoints.

| run | steps | exact | char | pos0 | pos1 | gate |
|---|---|---|---|---|---|---|
| `r38_curr8` — curriculum | 6k | 46.4% | 67.7% | 55.9% | 79.4% | 0.487 |
| **`r43_curr8_12k` — curriculum** | 12k | **73.6%** | 84.9% | **83.6%** | 86.2% | **0.828** |
| `r40_s3curr8` — shift-3 + curriculum | 6k | 46.6% | 66.6% | 60.8% | 72.5% | 0.475 |
| `r43_s3curr8_12k` — shift-3 + curriculum | 12k | 68.6% | 82.5% | 74.1% | 90.9% | 0.862 |
| `r37_f4` — 4-fact ceiling (reference) | 6k | 90.3% | 95.1% | 94.1% | 96.1% | 0.275 |

## Verdicts

- **The 46% was undertraining, not a ceiling. Confirmed.** Doubling steps takes the plain curriculum to **73.6%**
  (pos0 55.9 -> 83.6, gate 0.49 -> 0.83). Most of the gap to the 90.3% four-fact result was budget: the gate was
  still opening at 6k. 8 facts in text, out of context, answered from the store, now sits at three-quarters.
- **Plain curriculum overtakes shift-3+curriculum with enough training.** At 6k they tied (~46%); at 12k plain is
  **73.6%** vs shift-3's 68.6%. Once the curriculum has trained the disambiguation, shift-3's layout trick is not
  needed and slightly hurts — it splits the answer route (pos1 90.9 but pos0 only 74.1), where the plain run gets
  both high and balanced (83.6 / 86.2). Shift-3's value was as a *cold* fix (r37: 10 -> 31%); the curriculum
  subsumes it.

## Diagnosis

The language recall trajectory at 8 facts is now: **10.3% cold -> 46% curriculum@6k -> 73.6% curriculum@12k**,
against a 90.3% four-fact ceiling. Every gain is the gate opening further (0.21 -> 0.49 -> 0.83) as the model learns
to disambiguate the shared-delimiter key from a solved-easier start. This is the same curve MQAR climbed (r25), just
slower and noisier because the key shares a vocabulary with English prose. The remaining ~17 points to the ceiling
is plausibly more steps or a further ladder rung — untested.

## What this does NOT show

- Whether 8 facts reaches the 90% ceiling with still more steps (r44 pushes to 24k).
- One seed. r40 showed the curriculum swings ~1.5x across seeds (46 vs 69), so 73.6% is one draw on the high side of
  a wide band.
- The 12-fact ladder rung with an adequately-trained base (r40's was warm from a 6k base; r45 uses this 12k one).
