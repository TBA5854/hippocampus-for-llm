# r14 — results: a one-shot read, and a 4x store at 97.5% on facts seen once

## Numbers (single-occurrence facts, 8 facts, block-0 observation store, seed 0, `--n-steps 0 --beta 32`)

| run | M | slots filled | n_seg | batch x accum | steps | first >= 50% | first >= 95% | final | read mass (min / final) |
|---|---|---|---|---|---|---|---|---|---|
| r14_M512_oneshot | 512 | 384 | 4 | 32 x 1 | 4000 | 750 | 1250 | **100.0%** | 0.98 / 0.98 |
| **r14_M2048_oneshot** | 2048 | 1,920 | 16 | 8 x 4 | 6000 | 3500 | 5000 | **97.5%** (best 98.1%) | 0.81 / 0.94 |

Correct-slot (top-1) reads 100% at every eval of both runs.

```
M2048 recall  (steps 250..6000, every 250)
  2 3 5 3 3 3 3 2 12 19 19 28 36 53 59 59 75 94 94 96 97 98 98 98
read mass
  1.00 1.00 .96 .90 .85 .83 .81 .81 .87 .90 .92 .93 .93 .94 .94 .94 ...
```

## Verdicts

- **H14a (settling is unnecessary for exact-lookup keys) — confirmed.** One-shot read at 512 slots: 100%, same lift-off
  as the six-iteration settled read, and faster (8 min vs 11 for 4000 steps). GPU memory fell to 3.4 GB at batch 8 for
  the 4x store.
- **H14b (the 4x store holds >= 95%) — confirmed.** 97.5% final, >= 95% from step 5000. Lift-off later again: ~750 at
  1x, ~2250 at 2x, ~3500 at 4x (to 50%).

## Store-size scaling so far (observation store, facts seen once, effective batch 32)

| store | slots filled | distractor segments | first >= 50% | final |
|---|---|---|---|---|
| 1x | 384 | 2 | ~750 | 100% |
| 2x | 896 | 6 | ~2500 | 100% |
| 4x | 1,920 | 14 | ~3500 | 97.5% |

Recall at the end is not capped by store size; lift-off time grows with it.

## Two things seen along the way

1. **Read mass dipped, then recovered.** On the 4x store the one-shot read's weight on the correct slot fell from 1.00 to
   0.81 by step ~2000 while top-1 stayed at 100%, then rose back to 0.94 once the gate opened and recall lifted off.
   A mid-round plan to rerun at beta 64 (r15 staged) was not triggered: the run lifted off at the decision point. The
   dip is a pre-lift-off phenomenon here, not a fixed leak; beta 64 remains an option if a larger store stalls.
2. **The corpus is now the binding limit.** Final train LM loss 0.35 vs validation 2.59, validation rising from step
   ~3000. Each 4x example draws 16 segments of text, so 6000 steps x 32 examples x 16 x 128 tokens is ~390M tokens of a
   ~1.1M-character corpus: ~350 passes over tinyshakespeare. The language-model half memorises it. Recall survived, but
   any larger store, longer run, or natural-language claim needs a larger corpus. TinyStories is cached on the box.
