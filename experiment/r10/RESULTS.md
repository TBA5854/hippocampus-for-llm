# r10 — results: facts seen once, stored at the last block via a skip path, and persisted

## Numbers (single-occurrence facts, 4000 steps; key @ layer 0 position-free, frozen identity query, shift 1)

| run | value | facts | seed | best | final | first >= 50% |
|---|---|---|---|---|---|---|
| **r10_w5skip0** | **last block + block-0 skip** | 8 | 0 | 93.9% | **93.9%** | step 3000 |
| r10_obs_s1 | block 0 | 8 | 1 | 100.0% | **100.0%** | step 1250 |
| r10_obs_shuf | block 0, scrambled store | 8 | 0 | 4.4% | **2.3%** | never |
| r10_obs_f16 | block 0 | 16 | 0 | 100.0% | **100.0%** | step 750 |
| r10_obs_f32 | block 0 | 32 | 0 | 100.0% | **100.0%** | step 750 |

(r9_w0s1, block 0, 8 facts, seed 0: 100.0%.) Correct-slot reads 100% at every eval of every non-scrambled run.

```
w5skip0   3 3 3 5 3 3 8 5 6 20 46 74 85 91 93 94
```

**Offline, two processes, on r10_w5skip0's checkpoint** (`r10/offline/`, n=320): process 1 writes facts and distractors
into the store and exits; a fresh process loads it and answers a query segment containing no facts.

| | recall |
|---|---|
| in-process | 94.06% |
| **reloaded in a fresh process** | **94.06%** |
| fresh process, mismatched store file | 3.4% |

## Verdicts

- **H10a (skip path restores single-occurrence storage at the last block) — confirmed.** 2.3% (r9, last block alone) ->
  **93.9%**, still rising at the end of the budget. Late lift-off (~step 2250-3000 vs 750 for block 0).
- **H10b (observation store replicates) — confirmed.** Seed 1: 100%.
- **H10c (its control holds) — confirmed.** Scrambled 2.3%.
- **H10d (scales in facts) — confirmed.** 16 facts 100%, 32 facts (every key symbol) 100%, and lift-off is *not*
  slower with more facts (step 750 for both, vs 750-1250 at 8).

## Diagnosis

Two things are now separated cleanly:

1. **Storing observations removes the fact-count cliff.** The tiled prediction store (r6) failed at 16 and 32 facts
   because the value only carried what repetition let the model predict. The observation store has no such
   dependency: 32 single-occurrence facts at 100%. What remains as a limit is the key vocabulary (32 symbols), not
   the store.
2. **The last block can carry the observation if it is given a path.** A last-block value alone is 2.3%; adding the
   block-0 output lets the same last-block write reach 93.9%. The sum is slower to learn than block 0 alone — the
   last-block half is an unhelpful term the read path must learn to see past — which is what r11's 8000-step budget
   tests.

## Objective status

Read at the first layer, write at the last block, recall of facts **seen once**, from the model's own store with
nothing in the prompt, **persisted across processes: 94.06% == 94.06%**, mismatched store 3.4%. Single seed, 8 facts,
4000 steps for the last-block variant; r11 is running seeds, control, budget and scale for it.
