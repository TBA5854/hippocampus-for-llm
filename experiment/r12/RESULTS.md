# r12 — results: a 2x store learns much slower, but store size and batch size were both changed

## Numbers (single-occurrence facts, 8 facts, block-0 observation store, seed 0)

| run | M | n_seg | batch | steps | best | final | correct-slot reads | gate (final) |
|---|---|---|---|---|---|---|---|---|
| r9_w0s1 (ref) | 512 | 4 | 32 | 4000 | 100.0% | 100.0% (by step 1500) | 100% | 0.149 |
| **r12_M1024** | 1024 | 8 | 16 | 6000 | 44.7% | **43.1%** | 100% | 0.135 |
| r12_M2048 | 2048 | 16 | 8 | — | — | stopped at start, by design | — | — |

```
r12_M1024 (per 500 steps)   3 3 4 4 8 11 13 16 18 20 30 33 37 45 43
```

## Verdicts

- **H12a (2x store holds >= 95%) — not met: 43.1%**, still rising at the end of the schedule.
- **H12c (matching stays at 100%) — confirmed.** 100% correct-slot reads at every eval over 896 filled slots. Whatever
  slowed it is on the content/gate side, not retrieval.
- **H12b — not run.** Stopped at start: the 2x run changed store size and batch size together, and a 4x store at
  batch 8 would change both further. That is two confounds, not a measurement of store size.

## Diagnosis

The 2x run lifted off very late (~step 1750, vs ~750 at 512 slots) and climbed slowly (~7 points per 500 steps), with
the gate settling around 0.13. That profile matches **fewer examples per optimiser step** as well as it matches **more
distractor text in the store**: batch 16 is half the answer positions per step, and the answer position is the only
source of gradient that teaches the gate and `out` to use the read. The two cannot be separated from this run.

## Fix and next round

Gradient accumulation (`--accum`, commit 218bb4c) keeps the effective batch at 32 while each micro-batch still fits in
memory. r13 first checks accumulation reproduces the known 512-slot result (8 x 4), then reruns the 2x store at 16 x 2.
