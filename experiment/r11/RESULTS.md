# r11 — results: the last-block write is robust on facts seen once

## Numbers (single-occurrence facts, 8000 steps; key @ layer 0 position-free, frozen identity query;
## value = last block + block-0 skip, shift 1)

| run | facts | seed | best | final | lowest of last 8 evals | first >= 50% | first >= 99% |
|---|---|---|---|---|---|---|---|
| **r11_skip_s0** | 8 | 0 | 100.0% | **100.0%** | 100.0% | step 2250 | step 3750 |
| **r11_skip_s1** | 8 | 1 | 100.0% | **100.0%** | 100.0% | step 2250 | step 4250 |
| r11_skip_shuf | 8, scrambled | 0 | 4.4% | **2.8%** | 2.3% | never | never |
| **r11_skip_f32** | **32** | 0 | 100.0% | **100.0%** | 100.0% | step 1500 | step 3250 |

Chance 3.1%. Correct-slot reads 100% at every eval of every non-scrambled run.

```
(every other eval, i.e. one column per 500 steps)
skip_s0   3  6 11 23 66 86 95 99 99 100 100 100 100 100 100 100
skip_s1   4  3  6 14 62 85 92 97 99 100 100 100 100 100 100 100
skip_f32  4  3 49 94 94 98 99 100 100 100 100 100 100 100 100 100
shuf      3  3  3  4  3  4  3  3  2  3  3  4  2  3  4  3
```

## Verdicts

- **H11a (budget) — confirmed.** r10's 93.9% at 4000 steps was the budget: at 8000 the same config holds 100% from
  step 3750 to the end.
- **H11b (replicates) — confirmed.** Seed 1: 100%.
- **H11c (control) — confirmed.** Scrambled store 2.8%.
- **H11d (scales) — confirmed.** 32 facts: 100%, and it lifts off *earlier* than 8 facts (step 1500 vs 2250).

## What this settles

The objective's placement — **read at the first layer, write at the last block** — stores and recalls facts the
model saw exactly once, at 100% on two seeds and at 32 facts, with the scrambled store at chance. Its r10 checkpoint
already persisted across processes (94.06% == 94.06%, mismatched 3.4%); the r11 checkpoint should be re-tested.

Ingredients, each earned by a failure that came before it:

| ingredient | failure it fixes | round |
|---|---|---|
| key + read at layer 0 | layer-1 keys drift as block 0 trains; 1 seed in 3 | r4 -> r5 |
| position-free key and query | position similarity hijacks matching on untiled facts | r7 |
| query frozen at identity | the shared query is steered away from facts by text positions | r8 |
| value from the observed token (shift 1) | the prediction store holds only facts inferable from repeats | r6 -> r9 |
| block-0 skip into the last-block value | the last block alone no longer carries the observation | r9 -> r10 |
| 8000 steps | the skip path lifts off late | r11 |

## Lift-off note

The skip path is slower to start than a block-0 value (~step 2250 vs 750) but ends in the same place. More facts
*speed it up* (32 facts at step 1500): more answer positions per example, more gradient reaching the read path.

## Persistence (offline, two processes, on r11_skip_s0's checkpoint, n=320)

| | recall |
|---|---|
| in-process | **100.0%** |
| store saved, loaded in a fresh process, query segment has no facts | **100.0%** |
| fresh process, mismatched store file | 3.4% |

The objective in full, on facts seen once: read at the first layer, write at the last block, recall from the model's
own store with nothing in the prompt, persisted across processes, at 100%. Files in `r11/offline/`.
