# r6 — scale the layer-0 store: facts, then store size

**Contingent on r5_k0_s1.** If layer-0 keys fail on seed 1, this round is rewritten around whatever r5
shows is robust (gradient floor) before it launches.

## Ideate (from r5, mid-round)

`r5_k0` (key + read at layer 0, identity query init, value from the last block, shift 0): 8 facts
**100.0%** by step 1750, correct-slot reads 100% from step 250 onward with no decay. Layer-1 keys
decayed to 0% on 2 seeds of 3 as block 0 trained; layer-0 keys have no block-0 context to drift.

**Caveat carried forward.** A layer-0 key is token + position embedding, so matching is close to exact
token lookup. That is the right test of *storage and persistence inside the block*, and it removes the
drift. It is not evidence the store can retrieve by meaning: a paraphrased query would not match. A
contextual key that does not drift is a later round, stated here so the 100% is not over-read.

## Hypotheses

- **H6a — it replicates.** Seed 2 also reaches >= 95% (seed 1 is r5's job).
- **H6b — the control holds for this key space.** Scrambled store at chance: exact-lookup keys make a
  leak through the store's *structure* rather than its content worth ruling out explicitly.
- **H6c — facts scale.** 16 and 32 facts (32 is every key symbol) stay high. r4's taxonomy found every miss
  was key-vs-key confusion; with exact-lookup keys that confusion should vanish, so fact count should stop
  mattering until tiling thins (32 facts = 64 tokens, tiled only twice per segment).
- **H6d — store size scales.** 4x store (M=2048) with 4x content (n_seg 16: 14 distractor segments,
  ~1900 slots) holds recall. r4 found distractors never attract a read, so more of them should not hurt.

## Falsifiers

- Seed 2 or scrambled control off -> not robust / leak; stop scaling.
- 32 facts drops while 16 holds -> the limit is repetitions per fact in the fact segment, not store
  capacity; separate by adding fact-segment length.
- M=2048 drops -> distractor interference appears at scale after all (the correlated-key cliff predicts
  it gets harder as M grows).

## Runs (sequential; key+read @ layer 0, value @ last block, shift 0, identity query, seed 0 unless noted)

| order | name | change | ~time |
|---|---|---|---|
| 1 | r6_f8_s2 | seed 2 | 11 min |
| 2 | r6_f8_shuf | scrambled store | 11 min |
| 3 | r6_f16 | 16 facts | 11 min |
| 4 | r6_f32 | 32 facts | 11 min |
| 5 | r6_f8_M2048 | `--memory-size 2048 --n-seg 16` | ~45 min |
