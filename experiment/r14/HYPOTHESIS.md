# r14 — a one-shot read, so a 4x store fits the budget

## Ideate (from r13)

r13: gradient accumulation is faithful (512 slots at 8x4: 100%, same curve as batch 32), and the **2x store reaches
100% at effective batch 32** (r13_M1024_b16a2: lift-off ~step 2250, 96% at 2750, 100% at 3000). r12's 43% was mostly
the batch; store size delays lift-off but does not cap recall.

The 4x store (M=2048, 16 segments, 1,920 slots filled) is the next step, and its cost is the obstacle: the read's
`[B, S, M]` similarity is computed and kept for backward once per settling iteration, and graded settling runs 6 of
them (`n_steps` 3 + 2 extra + the final read). At 8 x 4 accumulation that is ~3 hours per 4000 steps.

With position-free token-embedding keys and a frozen identity query, matching is exact lookup: the queried token's
own slot has cosine 1.0. Attractor settling has nothing to refine. `--n-steps 0` reads once. Without the anneal the
forward temperature stays at `--beta`, so it is raised to 32 to keep the read sharp over ~2,000 slots (the backward
still uses `beta_grad` 4).

## Hypotheses

- **H14a — settling is unnecessary for exact-lookup keys.** One-shot read at 512 slots reaches ~100% like the settled
  read did.
- **H14b — the 4x store holds.** M=2048 at effective batch 32 with a one-shot read reaches >= 95% within 6000 steps.

## Falsifiers

- One-shot at 512 well below 100% -> settling (or the anneal to beta 64) matters; revert and budget the 4x run
  differently (fewer steps, or n_steps 1).
- 4x fails with matching at 100% -> the content/gate slowdown from distractor volume grows faster than 2x suggested;
  next lever: more answer positions per example (--n-facts 32 lifted off faster in r11) or answer weight.
- 4x OOM at batch 8 -> batch 4 x 8.

## Runs (single occurrence, 8 facts, block-0 observation store, seed 0, `--n-steps 0 --beta 32`)

| order | name | M | n_seg | batch x accum | steps |
|---|---|---|---|---|---|
| 1 | r14_M512_oneshot | 512 | 4 | 32 x 1 | 4000 |
| 2 | r14_M2048_oneshot | 2048 | 16 | 8 x 4 | 6000 |
