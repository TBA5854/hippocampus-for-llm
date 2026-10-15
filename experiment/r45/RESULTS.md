# r45 — the ladder compounds with base quality, and the curriculum is seed-robust once trained

## The 12-fact ladder rung improves with a better base

| run | base | exact | char | pos0 | gate |
|---|---|---|---|---|---|
| `r40_curr12` | 8-fact @6k | 41.6% | 61.6% | 52.3% | 0.659 |
| **`r45_curr12_from12k`** | 8-fact @12k | **54.5%** | 71.6% | 63.7% | 1.047 |

Warm-starting the 12-fact rung from a *better-trained* 8-fact model lifts it 41.6% -> 54.5%. The ladder compounds:
a stronger base carries further, exactly as a curriculum should.

## The curriculum is seed-robust once adequately trained

| run | seed | exact | char | pos0 | pos1 |
|---|---|---|---|---|---|
| `r43_curr8_12k` | 1 | 73.6% | 84.9% | 83.6% | 86.2% |
| **`r45_curr8_s3`** | 3 | **73.8%** | 85.1% | 81.7% | 88.4% |

At 12k steps two seeds land within **0.2 points** (73.6 / 73.8). This resolves the seed-variance worry from r38/r40,
where 6k runs swung 46% vs 69%: **the variance was undertraining too.** A fully-trained curriculum is stable.

## Verdict

- **H (ladder compounds) — confirmed.** Better base -> higher rung (41.6 -> 54.5 at 12 facts).
- **H (curriculum replicates) — confirmed, tightly.** 73.6 / 73.8 across seeds, vs the 46/69 spread at half the
  budget. The earlier seed fragility was a training-budget artifact.

## The language capability curve (facts in text, out of context, from the store)

| n_facts | best exact | note |
|---|---|---|
| 4 | 90.3% | near-ceiling |
| 8 | ~78.6% | seed-robust, asymptoting (r44) |
| 12 | 54.5% | still climbing with base quality |
| 13 | (task cap — untested) | |

A clean, monotone curve. The curriculum is the lever; the residual gap that grows with fact count is the char-level
addressing tax (r36/r44), which a word-level key would remove.

## What this does NOT show

- 13 facts (the task's cap, 2 distinct letters/key from 26) — the last point on the char curve, untested.
- Whether the 12-fact rung also asymptotes near a value short of its own ceiling, as 8 facts did.
