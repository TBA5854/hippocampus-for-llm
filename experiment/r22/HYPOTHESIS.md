# r22 — the two remaining ceilings: store size (16x) and key vocabulary (32 symbols)

## Ideate (from r20-r21)

r20 put the objective placement at 8x (4,064 slots) at 100%, persisted. r21 shows it is not seed luck (seed 1 at 100%
from the first eval) and that 32 facts fit the 8x store (100% from step 750, read mass 0.90 vs 0.97 at 8 facts). Two
ceilings are left, and they are different in kind:

1. **Store size.** Every doubling so far warm-starts more easily than the last (79.7% -> 96.2% at the first eval,
   4x -> 8x). 16x (M=8192, 64 segments, 8,128 slots) is the next step and the first that does not fit the compute
   budget comfortably: per-example cost is 4x r20's, so this runs at batch 2 x accum 16 for 500 steps (~100 min)
   rather than 1500.
2. **Key vocabulary.** `N_SYMBOLS = 32` in `mech/tasks.py` caps distinct keys, so "32 facts" means *every* key used
   exactly once — the fact count and the vocabulary are the same number, and neither can grow without the other.
   OBJECTIVE.md has named this the real limit since r11. Raising it to 128 separates the two for the first time.

## Code change (needed for run 2, tested before launch)

`--n-symbols N` (default 32) threading through `mqar_vocab`/`mqar_batch` instead of the module constant. Chance
becomes 1/N, so the control line moves: at 128 symbols chance is 0.8%, and a result that looks like "97%" is now
against a much harder baseline. Tests: vocab size scales, keys and values stay inside their ranges, and the existing
32-symbol default is unchanged (the whole round history depends on that default).

## Hypotheses

- **H22a — 16x reaches >= 95%** warm-started from the 8x checkpoint within 500 steps. Falsified if it opens below ~80%
  and does not climb: that would be the first sign of a real capacity limit rather than a lift-off cost, and the
  diagnostic is read mass (0.96-0.97 at every size so far; a drop means dilution).
- **H22b — 48 facts over a 128-symbol vocabulary reach >= 95%** at the 1x store. Falsified if recall falls while slot@1
  stays ~100% (content failure: the value cannot carry that many distinct symbols) — the r6 signature, which is how the
  fact-count cliff first showed up.

  At 1x, not 8x: the bigger vocabulary changes the model's shape (337 tokens vs 145), so `--init` cannot warm-start
  it, and a cold run at a large store does not lift off at all (r17). Running this cold at 8x would measure lift-off,
  not vocabulary. 1x trains cold to 100% (r18), so it isolates the question — at the cost of not testing both
  ceilings at once.

  48, not 64: each pair costs 2 tokens and `seg_len` is 128, so 64 facts would fill segment 0 exactly and leave no
  distractor text in the fact segment — a different (easier) task, not just a bigger one. 48 facts leave 32 tokens of
  text. This is also the first round whose chance line moves: 1/128 = 0.8%.

## Runs

| box | name | change |
|---|---|---|
| box1 | r22_16x | M=8192, n_seg 64, batch 2 x accum 16, 500 steps, eval every 125, `--init r20_warm_skipnorm_M4096.pt` |
| box2 | r22_1x_sym128_f48 | `--n-symbols 128 --n-facts 48` at the **1x** store, 4000 steps (chance 0.8%) |
| box2 | r22_1x_sym32_f8 | matched 32-symbol baseline at 1x, same seed — the line the first run is read against |

Deferred again: natural-language memory (the standing "what this does not show" item), and the learned write gate.
