# r9 — single-occurrence facts with a frozen identity query

## Ideate (from r7, r8)

Two rounds tried to ask "can the store hold a fact seen once?" and both stopped at matching:
r7 (layer-0 key with position) and r8 (position-free) both collapsed to 0% correct-slot reads. On the collapsed
checkpoint, an untrained identity query still matches 93.8% while the learned one — only 17% from identity —
matches 0%. The shared query projection is steered away by the text positions' gradient. With position-free
layer-0 keys matching is exact lookup, so the query is frozen at identity (`--freeze-query`).

## Hypotheses

- **H9e — matching holds to the end.** Correct-slot reads ~100% at every eval in every run. (Prerequisite.)
- **H9a — the prediction store cannot hold a single occurrence.** `w5s0` at chance with matching ~100%.
- **H9b — an observation store can.** `w0s1` (block-0 output at `v_j`) reaches high recall.
- **H9c — the last block still carries the observation.** `w5s1` reaches high recall; the objective's write
  position survives with observed content.
- **H9d — mid-stack.** `w3s1`.

## Falsifiers

- Matching still collapses -> the query is not the whole story (e.g. key side, store write); diagnose on the
  checkpoint before continuing.
- `w0s1` fails with matching intact -> the read/value path cannot deliver single-occurrence content even for
  an observation store; the next lever is the gate or the out projection.
- `w5s0` succeeds -> the prediction diagnosis is wrong.

## Runs (8 facts, single occurrence, 4000 steps, seed 0; key @ layer 0 position-free; frozen identity query)

| order | name | value from | shift |
|---|---|---|---|
| 1 | r9_w0s1 | block 0 output | 1 |
| 2 | r9_w5s1 | last block | 1 |
| 3 | r9_w5s0 | last block | 0 |
| 4 | r9_w3s1 | block 3 | 1 |
