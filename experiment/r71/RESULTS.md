# r71 — harden r70: RESULTS. **The controls REFUTE r70 — it was NOT chaining.**

## Numbers (best AND final)
| box | arm | recall |
|-----|-----|--------|
| box1 | pl_broken (2-hop, **chain CUT**) | **85.5%** (final 85.5) |
| box1 | pl_hop3 (3-hop) | 63.0% (final 62.7) |
| box2 | pl_recall_s1 (1-hop base, seed 1) | 97.3% |
| box2 | pl_hop2_s1 (2-hop, **seed 1**) | **7.8%** (final 0.5) |

## Verdict
Both hardening checks fail the r70 claim:
1. **Broken-chain = 85.5% = the intact 85%.** Cutting the link that the head needs to
   reach the tail changed nothing. A model that were truly chaining A->B->C would
   fall to chance here. It did not, so **r70's 85% was not chaining.**
2. **Seed 1 = 7.8% (collapsed to 0.5%).** The 85% did not replicate at a second seed,
   even though seed-1's 1-hop base was strong (97.3%). So it was not a robust
   capability either.

## The leak (task-construction bug, mine)
In `bpe_hop` the chain facts are placed in order with the tail token C as the value
of the LAST chain fact, and the query is ALWAYS the head A with target C. So C is
always the most-recently-written meaningful value and the target never varies in
role. A degenerate heuristic — "output the last / most-salient stored value" —
scores ~85% without reading the query at all, and the broken-chain control leaves C
in exactly that position, so it also scores 85%. seed 0 found this shortcut; seed 1
failed to fit and landed at chance. Neither is chaining.

## Corrected conclusion
**Multi-hop is NOT solved.** The shared store cannot chain (r68), and the per-layer
multistore does not chain either once the shortcut is removed (r71). r70's headline
is retracted (see the note prepended to r70/RESULTS.md).

## What r72 must do
Leak-proof the task before any multi-hop claim:
- **Multiple independent 2-hop chains** in seg0, shuffled into random order.
- **Query a RANDOM chain's head; target that chain's tail.** Then "return the last
  value" yields some other chain's tail (chance); only genuine chaining from the
  queried head succeeds.
- Re-run BOTH shared and per-layer multistore on the leak-proof task, with the
  broken-chain and shuffled-store controls. If both still fail, multi-hop needs the
  explicit inter-layer key transform (learned reindex) / thalamic relay — a real
  architectural addition, which becomes the next build.
