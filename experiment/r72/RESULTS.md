# r72 — leak-proof multi-hop: RESULTS. **Both stores fail at chance. Multi-hop is unsolved as-is.**

## Numbers (best AND final; leak-proof task, 4 shuffled chains, random queried chain)
| store | hop2 (leak-proof) | broken-chain control |
|-------|-------------------|----------------------|
| per-layer multistore (box1) | **0.2%** (final 0.0) | 0.2% (final 0.0) |
| shared (box2) | **0.5%** (final 0.5) | 0.0% |

## Verdict
With the "return the most-recent value" shortcut removed (query a random one of four
shuffled chains), **both stores collapse to chance** — and hop2 equals its own
broken-chain control in each case. This confirms r71: r70's 85% was entirely the
leak. Neither the single shared store nor the per-layer multistore performs genuine
2-hop composition.

Combined picture across the multi-hop rounds:
- r68: shared store, 2-hop ≈ 0 (retrieved value is not a clean key).
- r70/r71: per-layer multistore *looked* like 85% but it was a task leak; the
  broken-chain control matched it and a second seed collapsed.
- r72: leak-proof, both stores ≈ 0.

So single-hop recall, long-context, sparse read, CSA2, and update/recency all hold;
**composition does not come for free from either store topology.**

## Why — and the fix (r73)
Chaining needs hop L's retrieval to become a valid EXACT key for hop L+1. A raw
retrieved value is `scale·(norm(last_block) + norm(embed))` — it carries the value
token's embedding plus an equal-magnitude last-block component, so it does not land
on the token-embedding key manifold and the next hop's exact match never fires. No
store topology fixes this; it needs a **learned inter-layer key transform**: a
trainable map that re-lands a retrieval as a key (the "reindex" in DeepSeek terms,
the "thalamic relay" in the user's terms). r73 adds that map and re-runs the
leak-proof task with the broken-chain control. If it clears the control, chaining is
unlocked by the transform; if not, multi-hop needs a deeper change (e.g. training
the whole chain end-to-end with gradient through hops).
