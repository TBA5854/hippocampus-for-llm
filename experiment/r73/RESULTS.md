# r73 — learned reindex: RESULTS. **Negative — the linear reindex does not unlock chaining.**

## Numbers (best AND final; shared store, leak-proof task)
| arm | recall |
|-----|--------|
| reindex hop2 (seed 0, warm r70_sh_recall) | 0.5% (final 0.3) |
| reindex broken-chain control (seed 0) | 0.2% (final 0.0) |
| reindex hop2 (seed 1, base 96.4%) | 0.2% (final 0.0) |

## Verdict
The learned reindex sits at chance at both seeds, level with its broken-chain
control and with the plain shared store (r72, 0.5%). Adding a trainable map on the
retrieved value did **not** enable 2-hop composition.

## Why it failed (and what r74 changes)
The reindex is trained only through the CURRENT hop, and `hop_q` (the retrieval fed
to the next hop) is **detached**. So:
- the intermediate key is never directly supervised — there is no target saying
  "this retrieval should map to embed(B)";
- the only loss is the final answer, many layers downstream, and the detach severs
  the path from it back to the reindex and the earlier hop.
So the reindex gets essentially no gradient telling it how to re-land a retrieval.
An identity-initialised linear map with no useful signal stays useless.

r74 removes the detach: **end-to-end gradient through the hops** (`--hop-grad`), so
the final-answer loss backpropagates through every hop into the reindex and the read
path. Two arms: with reindex (the full fix) and without (does end-to-end gradient
alone suffice?), each with a broken-chain control.

## Standing status
Multi-hop remains unsolved after r68 (shared), r70/r71 (per-layer, was a leak), r72
(leak-proof both stores), r73 (linear reindex). Every genuine test is at chance. The
one lever untried is training the chain jointly (r74). If that also fails, the
retrieved value simply does not carry enough recoverable token identity, and
composition needs a different value representation or an explicit two-read decoder.
