# r7 — store what was observed, not what was predicted

## Ideate (from r6)

The winning configuration (key @ layer 0, value = last block at the key position, shift 0) stores the
model's *prediction* "next is `v_j`". r6 showed recall is 100% exactly where in-segment induction reached
its floor and chance where it did not, with matching ~100% throughout. So it only stores facts the model
could already infer from repeats. Real memory is mostly facts seen once.

New task: `--no-tile` writes each pair **once** at the start of segment 0 and fills the rest with text.
`v_j` cannot be predicted from context, so only a value holding the **observed** token can answer.

The value's layer and shift are independent of the key's (`--key-layer` / `--write-layer` /
`--value-shift`, shift applied to the value only). Key stays at layer 0 in every run (r5: no drift).

## Hypotheses

- **H7a — the current config cannot store a single occurrence.** `w5s0` on `--no-tile` stays at chance,
  with matching still ~100%. (Converts r6's inference into a direct measurement.)
- **H7b — an observation store can.** Value = layer-0 state at `v_j`'s position (`--write-layer 0
  --value-shift 1`, i.e. the token embedding itself) reaches high recall on single occurrences. This is the
  ceiling: a pure copy of what was seen.
- **H7c — the last block still carries the observation.** Value = last block at `v_j`'s position
  (`--write-layer 5 --value-shift 1`). The residual stream carries the embedding forward, so identity may be
  linearly recoverable by the learned `out` projection even though the head reads the block as a
  prediction. If this works, "write at the last block" survives with observed content.
- **H7d — a mid-stack value is the compromise.** `--write-layer 3 --value-shift 1`.

## Falsifiers

- `w5s0` succeeds -> something other than the prediction carries the answer; r6's diagnosis is wrong.
- `w0s1` fails -> the task or read path is broken for single occurrences regardless of value; fix that
  before interpreting the rest.
- `w5s1` fails while `w0s1` works -> the last block does not retain recoverable identity; the objective's
  write position has to move earlier, or the value needs a skip connection from an earlier layer.

## Runs (8 facts, single occurrence, 4000 steps, seed 0, key @ layer 0, identity query)

| order | name | value from | shift | purpose |
|---|---|---|---|---|
| 1 | r7_w0s1 | layer 0 | 1 | H7b ceiling / task sanity |
| 2 | r7_w5s1 | last block | 1 | H7c — keeps the objective's write position |
| 3 | r7_w5s0 | last block | 0 | H7a — current config, predicted to fail |
| 4 | r7_w3s1 | block 3 | 1 | H7d |
