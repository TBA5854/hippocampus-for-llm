# r20 — one more curriculum step: warm-start 8x from the 4x model

## Ideate (from r19)

Warm-starting the 4x store (M=2048) from the 2x model hit 99.1% by step 750 and 100% from step 1500, persisted at 100%,
and its scrambled control was at chance. From scratch, the same 4x run never lifted off. The diagnosis is that store size
costs lift-off, not capacity: read mass stayed at 0.97 from the first eval. If that holds, 8x (M=4096, 32 segments,
4064 slots filled per example) should need only another warm start.

## Hypotheses

- **H20a — warm start from `r19_warm_skipnorm_M2048` reaches >= 95% at 8x** within 1500 steps.
- **H20b — scrambled store with the same init is at chance.**

## Falsifiers

- Step 250 near chance with read mass well below 0.97 -> doubling the store *does* dilute the transferred read at 8x.
  That would be a capacity or interference limit, not lift-off, and the next lever would be beta (sharper one-shot read)
  or key separation, not more curriculum.
- Control above chance -> the transferred weights answer without the store.

## Budget

At M=2048, batch 16 used 5.4 GB and ~45 min for 4000 steps. Per-example memory and compute scale with n_seg × M, 4x
here, so batch 4 x accum 8 (effective 32, ~5.4 GB). Each step costs ~4x the compute, so steps are cut to 1500 (r19 was at
99% by 750), about 1 h per run.

## Runs (as r19, but M=4096, 32 segments, batch 4 x accum 8, 1500 steps, --init r19_warm_skipnorm_M2048)

| order | name | store |
|---|---|---|
| 1 | r20_warm_skipnorm_M4096 | real |
| 2 | r20_warm_skipnorm_M4096_shuf | scrambled |
