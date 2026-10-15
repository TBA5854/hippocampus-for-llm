# r13 — store size vs batch size

## Ideate (from r12, mid-round)

r12_M1024 (2x store, 8 segments, batch 16) climbed slowly: 3% until step ~1750, 17.5% at step 3500, gate ~0.12. The
512-slot observation store at batch 32 reached 100% by step 1500. r12 changed two things at once — the store doubled
and the batch halved (memory scales with segments x slots, so it had to). r12's second run (4x store at batch 8)
would be doubly confounded and was stopped before it started.

`--accum k` (commit 218bb4c) runs k micro-batches per optimiser step, holding the effective batch at 32 while memory
per micro-batch stays small.

## Hypotheses

- **H13a — accumulation is faithful.** 512 slots, 8 x 4 accumulation reaches ~100% like batch 32 did (r9_w0s1).
  If not, accumulation itself changes the dynamics and cannot be used to separate the two.
- **H13b — the r12 slowdown was batch, not store.** 2x store at effective batch 32 (16 x 2) reaches >= 95%.

## Falsifiers

- `M512_b8a4` well below 100% -> accumulation is not equivalent here; fall back to comparing stores at equal
  *small* batch instead.
- `M1024_b16a2` stays slow like r12_M1024 -> store size (more distractor text in the store) genuinely slows it;
  diagnose with the miss taxonomy on its checkpoint.

## Runs (single occurrence, 8 facts, 6000 steps, block-0 observation store, seed 0)

| order | name | M | n_seg | batch x accum |
|---|---|---|---|---|
| 1 | r13_M512_b8a4 | 512 | 4 | 8 x 4 |
| 2 | r13_M1024_b16a2 | 1024 | 8 | 16 x 2 |
