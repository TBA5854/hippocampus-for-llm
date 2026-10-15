# r9 — results: a store that holds facts seen once, and only from early in the stack

## Numbers (single-occurrence facts `--no-tile`, 8 facts, 4000 steps, seed 0)

Key @ layer 0, position-free, query frozen at identity. Only the value source changes.

| run | value = | best | final | correct-slot reads (all run) | gate (final) |
|---|---|---|---|---|---|
| **r9_w0s1** | block-0 output at `v_j` (observed) | 100.0% | **100.0%** | 100% | 0.149 |
| r9_w3s1 | block-3 output at `v_j` | 25.0% | **25.0%** | 100% | 0.076 |
| r9_w5s1 | last block at `v_j` | 4.4% | **2.3%** | 100% | 0.001 |
| r9_w5s0 | last block at the key (prediction) | 4.4% | **2.3%** | 100% | 0.001 |

Chance 3.1%.

```
w0s1   3  3 76 97 100 100 100 100 100 100 100 100 100 100 100 100
w3s1   3  3  3  4   3   3   4  11  10   4   8  15  13  23  23  25
w5s1   3  3  3  4   3   3   4   3   3   3   4   3   4   4   2   2
w5s0   3  3  3  4   3   3   4   3   3   3   4   3   4   3   2   2
```

## Verdicts

- **H9e (matching holds to the end) — confirmed.** 100% correct-slot reads at every eval of every run. Freezing the
  identity query ends the collapse r7 and r8 hit. Every difference below is content, not matching.
- **H9b (an observation store holds a single occurrence) — confirmed.** **100%**, lift-off by step 750.
- **H9a (the prediction store cannot) — confirmed**, directly: 2.3%. r6's inference ("only facts the model could
  already predict are stored") is now a measurement on facts that cannot be predicted.
- **H9c (the last block still carries the observation) — falsified.** 2.3%.
- **H9d (mid-stack)** — partial: 25%, still climbing slowly at the end.

## Diagnosis

With matching held fixed at 100%, recall reads off how much of the *observed* token the stored state still carries
in a form the read path can use: **block 0 -> 100%, block 3 -> 25%, block 5 (last) -> 2%.** The residual stream
keeps the embedding in principle, but by the last block it is not recoverable through `out` + gate + the rest of the
stack, and the gate learns to stay shut (0.001). The deeper the write, the more the state is about what comes next
and the less about what was seen.

This inverts the original design intuition ("write late, the representation is richest there") for one important
case: the richest late representation is a *prediction*, and a prediction of an unpredictable fact is empty.

## What it changes

- **First result for facts seen once**: in-block memory, read at the first layer, stores and recalls single-occurrence
  facts at 100%, with the write taken from block 0. The limit recorded in `OBJECTIVE.md` (prediction caching) is
  overcome — but not at the last block.
- **The objective's write position is in tension with single-occurrence storage.** r10 tests the direct compromise:
  write at the last block with a skip connection from block 0 (`--value-skip-layer 0`), so the store is still written
  at the end of the pass but carries the observation.

## Robustness still owed for the block-0 store (in r10)

Seed replication, scrambled-store control, and 16/32 single-occurrence facts. Persistence re-test on its checkpoint.
