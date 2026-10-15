# r100 — a growable store (append like a KV cache) removes the r51 ring cliff

**Growing the store by appending — amortized doubling, "grow like a KV cache" — instead of
a fixed ring that laps holds recall out to 199 windows (25k tokens, 25,273 slots) where the
fixed ring collapses to 0% at 7 windows. memory_size becomes just the initial capacity; the
store is now unbounded.**

## Context-bridge recall vs bridge distance (arch2_d512_L2, plain store, M_init=512, dense read)
| windows (tokens) | slots | fixed ring | grow |
|------------------|-------|-----------|------|
| 3 (384)    | 381   | 100.0% | 100.0% |
| 7 (896)    | 889   | **0.0%** | 96.9% |
| 15 (1920)  | 1905  | 0.0% | 84.4% |
| 31 (3968)  | 3937  | 0.0% | 82.8% |
| 63 (8064)  | 8001  | 0.0% | 76.6% |
| 127 (16256)| 16129 | 0.0% | 64.1% |
| 199 (25472)| 25273 | 0.0% | **62.5%** |

## Reading
- **The cliff is gone.** The fixed ring holds only while `slots <= M` (3 windows at M=512),
  then laps and overwrites seg-0's facts -> 0% (the r51 cliff). The growable store appends
  (doubling capacity as needed, up to 25,273 slots here) and never laps, so seg-0's facts
  survive arbitrarily far — recall stays non-zero out to 199 windows.
- **The residual decay is the DENSE read, not the growth.** Recall falls 100 -> 62.5% as the
  store grows because this probe uses the dense graded read over ALL slots, and distractor
  mass grows with slot count — exactly r62's finding (dense decays with store size, top-1 is
  immune). With the sparse top-1 read (r62/r64) the growable store would hold near-flat, since
  the token pool touches O(1) slots regardless of size. Growth removes the cliff; the sparse
  read removes the decay; the two compose.

## Verdict
`--store-grow` makes the store an unbounded, append-only KV-cache-style memory: `memory_size`
is only the seed capacity, and the r51 `slots==M` cliff disappears. Combined with the sparse
candidate-pool read (O(1) slots, r64) for flat recall and the sleep loop (r27) to consolidate
old slots into weights and reclaim, this is the full lifelong-memory container. Plain write
path only; the gated store already bounds itself to O(facts). Folded into the paper.
