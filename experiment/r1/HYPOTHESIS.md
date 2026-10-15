# r1 — lock the baseline on the objective's architecture

## Ideate

The only controlled positive result (`st_bg4`, 100% 2-fact MQAR) writes at block 4 of 6. The objective
writes at the last block, block 5, right before `ln_f` and the head. Everything later is built on that
placement, so it has to be confirmed before anything else changes.

## Hypothesis

Moving the write from block 4 to block 5 preserves 100% recall. The last block's state is the most
complete representation of a position, and a read at layer 1 leaves the whole stack to use it.

## Prediction and falsifier

- `w4` reproduces `st_bg4`: ~100%. **If it does not, the environment has drifted** and no other number
  this round means anything.
- `w5` >= 90%. **Falsified below that**, which would mean the last block is specialised for next-token
  prediction and is a worse thing to store than a mid-stack state. That would be a finding in itself and
  a fork: write-late would need to mean "late but not last".
- Both controls at `w5` land near chance (~3-4%).

## Runs (one variable changed; everything else is `st_bg4` verbatim)

| name | change | expect |
|---|---|---|
| r1_w4 | none (reproduction) | ~100% |
| r1_w5 | `--write-layer 5` | >= 90% |
| r1_w5_shuf | `--write-layer 5 --shuffle-store` | chance |
| r1_w5_off | `--write-layer 5 --hc off` | chance |

Controls are at matched `beta_grad`, `dg_bypass`, `seed`, `answer_weight`, `n_heads`.
