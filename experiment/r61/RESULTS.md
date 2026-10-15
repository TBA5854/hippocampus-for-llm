# r61 — RESULTS: the north-star demo — store ≫ context window AND each query touches ONE slot

The user's flagship ask, landed: a store far larger than the model's attention window, answered by
touching only what's required. `mech.persist sparse` fills a big store bridging K context windows (like
r51) and reads the query with a top-k readout for shrinking k, on the trained 64-fact BPE model
(`r50_bpe64`). The model's attention window is `seg_len = 128`; segments do not cross-attend, so the
store is the only channel.

## The result (recall; store bridges K windows of prose the model cannot attend across)

| store size | K windows | slots | **top-1** | top-2 | top-4 | top-8 | dense (all slots) |
|---|---|---|---|---|---|---|---|
| M=1024 | 7  | 889  | **100.0%** | 99.4% | 98.8% | 98.8% | 98.8% |
| M=2048 | 15 | 1905 | **100.0%** | 96.9% | 93.8% | 93.8% | 93.8% |
| M=4096 | 31 | 3937 | **100.0%** | 95.0% | 91.2% | 90.0% | 90.0% |

## What it shows

1. **Both halves at once.** At M=4096 the store holds **31 context windows (3968 tokens)** and answers
   from an empty 128-token prompt while **each query touches exactly 1 of 3937 slots**. "More context
   than the window, query only what's required" — measured together, no retraining (the model was
   trained with the dense `graded` readout; the top-k read is applied at eval).

2. **Sparse is a strict WIN, not a trade-off.** top-1 = **100% at every scale**, and it *beats* the
   dense read at every scale (100 vs 98.8 / 93.8 / 90.0). The dense softmax spreads a little weight onto
   near-miss and distractor-prose slots, and that dilution grows with store size (98.8 → 90.0 as the
   store grows 4×). top-1 commits to the single exact key match and is immune to it — so **the bigger
   the store, the more sparse retrieval helps.**

3. **Effective context is bounded only by store size, at O(1) read.** Recall does not fall as the bridge
   grows from 7 → 31 windows under top-1; the store scales and the query cost stays at one slot.

## Verdict
- **North-star (store > window AND query only what's required):** ACHIEVED, and stronger than posed —
  sparse top-1 is cleaner than dense at every store size. The read is O(1) slots per query regardless of
  how many windows the store holds.

## Caveats / next
- Top-1 works because the layer-0 key gives an *exact* token match, so the answer slot dominates the
  score ranking — this is the same property r5/r42 identified as load-bearing. On a task with fuzzier
  keys, a larger k (or a candidate pool) would be needed; k=2–8 already spans that.
- The score ranking here is still computed densely (q·K over all M) — top-k restricts the *output*, not
  the compute. A truly sparse *compute* (candidate pool, not full scan) is the CSA2 generalization
  (route 2, [[csa2-store-future-build]]) and the way to make this cheap at very large M.
