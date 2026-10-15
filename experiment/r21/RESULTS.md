# r21 — results: 8x is not seed luck, and it holds 32 facts

## Numbers (TinyStories, single-occurrence facts, one-shot read beta 32, key @0 position-free, frozen identity query,
## write @ last block + normalised block-0 skip, M=4096, 32 segments, 4 x accum 8, 1500 steps,
## --init experiment/r19/ckpt/r19_warm_skipnorm_M2048.pt)

| run | box | change | 250 | 500 | 750 | 1000 | 1250 | final | mass | slot@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| **r21_8x_seed1** | box1 | `--seed 1` | 100% | 100% | 100% | 100% | 100% | **100.0%** | 0.97 -> 0.96 | 100% |
| **r21_8x_f32** | box2 | `--n-facts 32` | 98.8% | 97.5% | 100% | 100% | 100% | **100.0%** | 0.90 -> 0.89 | 100% |

Chance 3.1%. Both runs: alignment 100% at offset 0 at every eval; match-miss 0% throughout. The 32-fact run's only
errors were content (1.2% at step 250, 2.5% at step 500), gone from step 750 on. Val ppl 2.02 (seed 1) and 2.06 (32
facts).

## Verdicts

- **H21a (seed 1 reaches >= 95% at 8x) — confirmed.** 100% at the *first* eval, above seed 0's 96.2%. The store-size
  curriculum is not carrying seed-specific structure.
- **H21b (32 facts reach >= 95% at 8x) — confirmed.** 100% from step 750.

## Diagnosis

Read mass separates the two runs cleanly: 0.97 at 8 facts, 0.90 at 32 facts, each flat across the whole run. Four times
as many keys competing costs ~7 points of mass on the correct slot and delays nothing else — recall is still 100%, and
matching never misses. So at 8x the store is nowhere near its interference limit.

The fact-count axis is now exhausted for a different reason: `N_SYMBOLS = 32` in `mech/tasks.py` caps distinct keys, so
"32 facts" is *every key in the vocabulary used exactly once*. Fact count and key vocabulary have been the same number
in every round since r1. r22 adds `--n-symbols` to separate them.

## Where the objective placement stands (facts seen once, TinyStories)

| axis | established |
|---|---|
| store size | 1x, 2x, 4x, **8x** (4,064 slots) at 100%, via a size curriculum |
| seeds | 0 and 1 at 8x; 0, 1, 2 at 1x (r5, r11) |
| facts | 8, 16, 32 — but 32 = the whole key vocabulary |
| persistence | 4x and 8x: reloaded in a fresh process == in-process (100%), mismatched store at chance |
| controls | scrambled store at chance at every size tested |

## Next

r22, running: 16x (M=8192, 64 segments) warm-started from the 8x checkpoint on box1, and the key-vocabulary question on
box2 (`--n-symbols 128`, 48 facts, at the 1x store where cold training works).
