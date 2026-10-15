# r17 — results: a normalised skip does not lift off at 4x either; the 4x control is at chance

## Numbers (single-occurrence facts, 8 facts, TinyStories, one-shot read beta 32, M=2048, 16 segments, 16 x 2, seed 0)

| run | value | steps | best | final | gate (final) | read mass min / final | correct-slot reads |
|---|---|---|---|---|---|---|---|
| r17_skipnorm_M2048 | mean\|skip\| * (unit(last) + unit(block 0)), write @ last block | 12000 | 5.9% | **4.4%** | 0.018 | 0.46 / 0.46 | 100% |
| r17_obs_M2048_shuf | block 0, **scrambled store** | 6000 | 5.9% | **5.9%** | 0.012 | — | — |

Chance 3.1%.

## Verdicts

- **H17a (normalising the halves lets the last-block write lift off at 4x) — falsified.** Chance for all 12000 steps.
  What normalising did change: the gate stayed open (~0.02-0.03) instead of closing as in r16 (0.020 -> 0.010). What it
  did not change: read mass fell on the same curve as r16 (0.79 at step 2000 in both, 0.46 at the end), and recall never
  moved.
- **H17b (4x scrambled control at chance) — confirmed, with a note.** Final 5.9% against 3.1% chance. The eval uses the
  same 320 queries every time and the run moved between 1.6% and 5.9% with no trend, so this is eval noise on a fixed
  set rather than a leak; for comparison the real 4x store reached 100% (r15).

## Diagnosis

The estimate that the skip path at 4x might lift off near step ~11000 did not hold within this schedule. Read mass on the
correct slot fell steadily without the recovery every successful run showed at lift-off. Together with r15 (the
block-0 store at 4x lifted off and mass recovered), the pattern is consistent with: without early memory use there is no
pressure keeping key-symbol embeddings separated from text tokens, the read dilutes, and lift-off gets harder over time.

r18 was queued to find where along store size the skip path stops working, on the same corpus.
