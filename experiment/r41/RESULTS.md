# r41 — placement: reading at every layer does not beat reading at the start

The user's long-standing ask ("r start w end vs r/w at evry layer, both are gated"), read-side rung. Write stays at
the last block; the read is either at layer 0 only or at all six layers, one shared store, 4 facts, matched.

| run | exact | char | pos0 | per-layer gates |
|---|---|---|---|---|
| `r41_read0` — read @0 only | **90.3%** | 95.1% | 94.1% | 0: 0.275 |
| `r41_readall` — read @every layer | 88.3% | 93.9% | 92.8% | 0:0.24 1:0.21 2:0.23 3:0.24 4:0.20 5:0.13 |

## Verdict

- **H41a — the early read is sufficient; reading everywhere does not help. Confirmed** (with a nuance). Read-at-all
  is 88.3% against read-at-0's 90.3% — a tie within seed noise, if anything slightly worse. The nuance: the deep
  gates do **not** stay shut as strictly predicted — all six open to 0.20-0.24, with layer 0 highest (0.24) and the
  last block lowest (0.13). So the model *is* willing to read at depth; it just gains nothing by it, and the spread
  of gate mass across six heads slightly dilutes the layer-0 read that was carrying the task.

## Diagnosis

This confirms the objective's placement choice was right, and cheaply. The store holds position-free layer-0 keys
(near-exact token lookup, r31d), and the useful read is the one that queries in that same space — at layer 0. Deeper
reads query contextualised states that no longer match the stored keys exactly, so they retrieve nothing worth the
gate they open. "Read early, write late" is not just a reasonable default; reading late as well measurably fails to
add anything and marginally hurts.

This settles the read half of the placement question. The write half (writing at every layer, one shared store vs
one per layer) needs plumbing — `write_layer` is a single int — and is the remaining piece.

## What this does NOT show

- One seed. The 2-point gap is within the spread this task shows, so "tie" is the honest reading, not "read-at-0 is
  strictly better".
- Writes at every layer: unbuilt, deferred.
