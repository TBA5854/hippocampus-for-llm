# r13 — results: a 2x store reaches 100%; r12's slowdown was mostly the batch

## Numbers (single-occurrence facts, 8 facts, block-0 observation store, seed 0, 6000 steps)

| run | M | n_seg | batch x accum | effective batch | first >= 50% | first >= 99% | final |
|---|---|---|---|---|---|---|---|
| r9_w0s1 (ref, 4000 steps) | 512 | 4 | 32 x 1 | 32 | ~750 | ~1250 | 100.0% |
| **r13_M512_b8a4** | 512 | 4 | 8 x 4 | 32 | 750 | 1250 | **100.0%** |
| r12_M1024 (ref) | 1024 | 8 | 16 x 1 | 16 | — | — | 43.1% |
| **r13_M1024_b16a2** | 1024 | 8 | 16 x 2 | 32 | 2500 | 3000 | **100.0%** |

Correct-slot reads 100% at every eval in both runs.

```
r13_M1024_b16a2   ... 1500: 5  1750: 11  2000: 9  2250: 26  2500: 68  2750: 96  3000: 100 ... 6000: 100
r12_M1024         ... 1500: 4  1750: 8   2000: 5  2250: 11  2500: 13  2750: 8   3000: 16  ... 6000: 43
```

## Verdicts

- **H13a (accumulation is faithful) — confirmed.** 8 x 4 at 512 slots reproduces batch 32: 54% at step 750, 97.5% at
  1000, 100% from 1250. So accumulation can be used to hold effective batch fixed as the store grows.
- **H13b (the r12 slowdown was batch, not store) — confirmed, with a residual.** Same 2x store, same steps, effective
  batch 16 -> 32: **43.1% -> 100%**. Store size still delays lift-off (~step 2250 vs ~750 at 512 slots) but does not cap
  recall.

## Also seen

- A brief instability at step 1500 (recall 8.8% -> 4.7%, query-segment loss 0.06 -> 0.13) that recovered by the next
  eval.
- Validation LM loss on distractor text rises steadily after lift-off (1.40 at step 3000 -> 1.82 at 6000) at micro-batch
  16. Recall is unaffected (100% throughout); it is language-model overfitting on tinyshakespeare, noted not chased.

## Next

The 4x store (M=2048, 16 segments) at 8 x 4 costs ~3 h per 4000 steps because graded settling computes and keeps the
`[B, S, M]` read six times. r14 reads once (`--n-steps 0`, `--beta 32`) — exact-lookup keys give settling nothing to
refine — validated at 512 slots first.
