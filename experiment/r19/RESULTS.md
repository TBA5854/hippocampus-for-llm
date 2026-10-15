# r19 — results: a store-size curriculum gets the objective placement to 4x

## Numbers (TinyStories, single-occurrence facts, 8 facts, one-shot read beta 32, key @0 position-free, frozen identity
## query, write @ last block + normalised block-0 skip, M=2048, 16 segments, 16 x 2, 4000 steps, seed 0,
## --init experiment/r18/ckpt/r18_skipnorm_M1024.pt)

| run | store | step 250 | first >= 95% | best | final | gate 250 -> final | read mass | correct-slot reads |
|---|---|---|---|---|---|---|---|---|
| **r19_warm_skipnorm_M2048** | real | 79.7% | 750 | 100.0% | **100.0%** | 0.184 -> 0.316 | 0.97 throughout | 100% |
| r19_warm_skipnorm_M2048_shuf | scrambled | 3.4% | — | 5.9% | **2.8%** | 0.173 -> 0.123 | — | — |

Chance 3.1%. The real run held 100% from step 1500 to 4000, apart from a single 99.7% at 2250, with val ppl 1.99.
Content-miss and match-miss were 0% by the end, and alignment was 100% at offset 0.

Persistence on the real run's checkpoint (`mech.persist`, two processes, n=40 queries, 1905 slots per example, 33.6 MB
per store file):

| in-process | reloaded in a fresh process | mismatched store |
|---|---|---|
| 100% | **100%** | 2.5% |

## Verdicts

- **H19a (warm start reaches >= 95% at 4x) — confirmed.** 79.7% at the first eval, 99.1% by step 750, 100% from 1500.
  From scratch the same configuration was at chance through 12000 steps (r17).
- **H19b (the warm-started model reads the store's content) — confirmed.** The same initialisation with a scrambled store
  is at chance from its first eval. Its gate, inherited open at 0.173, closes steadily as training learns that a
  scrambled store is useless. The real run's gate opens instead (0.184 -> 0.316).

## The objective placement across store sizes (TinyStories)

| store | from scratch, normalised skip | warm start from the size below |
|---|---|---|
| 1x | 100% (plain skip, r18) | — |
| 2x | 99.4% (r18, lift-off ~5250) | — |
| 4x | chance through 12000 (r17) | **100%** (r19, >= 95% by 750), persists at 100% |

## Diagnosis

The 4x failure was a lift-off problem, not a capacity problem. A model that has already learned to use the read at 2x
carries it over to 4x with almost no adaptation. The read mass on the correct slot stays at 0.97 from the first eval,
so the larger store does not dilute a read that already exists. What the cold runs lacked was the early pressure that
keeps key embeddings apart (r17's diagnosis). A warm start supplies that pressure from step 0.

The objective as stated now holds at 4x: read at layer 0, write at the last block before the head, facts seen once and
out of context, the store persisted and reloaded in a fresh process, and controls at chance. The prediction for 8x is
that it needs one more curriculum step rather than a new mechanism. r20 tests that.

## Next

r20 extends the curriculum to 8x: warm-start M=4096 (32 segments) from `r19_warm_skipnorm_M2048`, with a matched
scrambled control. Memory scales with n_seg × M, so this uses batch 8 x accum 4. A second seed of the 2x -> 4x step
checks that the curriculum result is not tied to seed 0.
