# r10 — a skip path for the last-block write; lock in the observation store

## Ideate (from r9)

On single-occurrence facts (each pair seen once), with position-free layer-0 keys and a frozen identity query:

| value | r9 final |
|---|---|
| block-0 output at `v_j` (observed) | **100.0%** |
| last block at `v_j` (shift 1) | 2.3% |
| last block at the key (shift 0, the prediction) | 2.3% |
| block 3 at `v_j` | (see r9/RESULTS.md) |

Matching was 100% in every run, so every failure is content. The prediction store cannot hold a fact seen once
(r6's inference, now measured), and by the last block the observed token is no longer recoverable through the
read path. An early-block value works.

Two directions follow:

1. **Keep the write at the last block and give it a skip connection** (`--value-skip-layer 0`): value =
   last-block state + block-0 output at `v_j`. The store is still written at the end of the forward pass, as the
   objective specifies, but carries the observation.
2. **Make sure the observation store is real, not a lucky seed or a leak**, and find where its fact-count cliff is.

## Hypotheses

- **H10a — the skip path restores single-occurrence storage at the last block.** `w5skip0` reaches high recall.
- **H10b — the observation store replicates.** Seed 1 reaches >= 95%.
- **H10c — its control holds.** Scrambled store at chance.
- **H10d — it scales in facts.** 16 and 32 single-occurrence facts stay high. Unlike the tiled prediction store,
  content no longer depends on repeats, and matching is exact lookup, so fact count should not matter until the
  32-symbol vocabulary is exhausted (32 facts = every key symbol).

## Falsifiers

- `w5skip0` fails -> adding the last block's state corrupts the observed token (it dominates the sum); try
  a learned mix or the skip alone.
- Seed 1 fails / scrambled above chance -> the 100% is not robust or not content.
- 16/32 drop -> a capacity or interference limit specific to single occurrences.

## Runs (single occurrence, 4000 steps; key @ layer 0 position-free, frozen identity query, shift 1)

| order | name | value | facts | seed |
|---|---|---|---|---|
| 1 | r10_w5skip0 | last block + block-0 skip | 8 | 0 |
| 2 | r10_obs_s1 | block 0 | 8 | 1 |
| 3 | r10_obs_shuf | block 0, scrambled store | 8 | 0 |
| 4 | r10_obs_f16 | block 0 | 16 | 0 |
| 5 | r10_obs_f32 | block 0 | 32 | 0 |
