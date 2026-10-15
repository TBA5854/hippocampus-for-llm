# r8 — observed vs predicted values on single-occurrence facts, position-free keys

## Ideate (from r7)

r7 asked whether the store can hold a fact seen **once** (`--no-tile`), but its ceiling run never got to the
value question: layer-0 keys carry position, and with the facts at positions 0-15 and the query at 126, the
read switched to position-matched text (0% on fact keys from step 1250). `--key-pos-free` removes position
from key and query. r8 is r7's design, unchanged except for that flag.

## Hypotheses (as r7)

- **H8a — the prediction store cannot hold a single occurrence.** `w5s0` (value = last block at the key,
  shift 0) at chance, matching ~100%.
- **H8b — an observation store can.** `w0s1` (value = block-0 output at `v_j`) reaches high recall.
- **H8c — the last block still carries the observation.** `w5s1` (last block at `v_j`) reaches high recall;
  the objective's write position survives with observed content.
- **H8d — mid-stack.** `w3s1`.
- **H8e — position-free matching holds all run.** Correct-slot reads stay ~100% for the whole run in every
  variant (r7's failure was a collapse at step 1250, so this must be checked to the end).

## Falsifiers

- Matching collapses again -> something other than position embedding drives it; stop and diagnose.
- `w0s1` fails with matching intact -> single-occurrence recall is failing on the value/read side even for
  the observation store; the task or read needs work before value placement means anything.
- `w5s0` succeeds -> the prediction diagnosis is wrong.

## Runs (8 facts, single occurrence, 4000 steps, seed 0, key @ layer 0 position-free, identity query)

| order | name | value from | shift |
|---|---|---|---|
| 1 | r8_w0s1 | block 0 output | 1 |
| 2 | r8_w5s1 | last block | 1 |
| 3 | r8_w5s0 | last block | 0 |
| 4 | r8_w3s1 | block 3 | 1 |
