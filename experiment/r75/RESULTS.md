# r75 — pointer loop: RESULTS. **Negative. Closes the implicit multi-hop line.**

## Numbers (best AND final; shared store, leak-proof task)
| arm | hop2 | broken control |
|-----|------|----------------|
| pointer (seed 0) | 0.2% | 0.0% |
| pointer (seed 1, base 96.4%) | 0.0% | 0.2% |

Decoding the retrieval onto the token-embedding manifold and re-querying with that
clean key — trained end-to-end — still gives chance. Even projected to the manifold,
hop-1's retrieval does not land on the intermediate token reliably enough, and the
only training signal (the final answer) never teaches the decode *what* to point at.

## The line this closes
Every **implicit** route to multi-hop is now exhausted, all at chance:
- r68 shared read · r70/71 per-layer (was a leak) · r72 leak-proof both stores ·
  r73 learned reindex · r74 end-to-end gradient · r75 decoded pointer.

The consistent lesson: **within-forward read-chaining is the wrong mechanism.** A
single store lookup per pass, keyed by the current token, cannot compose — no key
transform, gradient path, or decode fixes it, because composition is not a lookup
problem.

## The reframe (why r76 changes organs)
Composition is a **working-memory** operation, not a long-term-lookup operation: you
retrieve B, **hold it in mind**, and use it to fetch C — retain, don't re-perceive.
That needs (a) writing a *computed intermediate* back to memory, and (b) a
**controller** that decides, per step, what to read and whether to write-new / merge
/ update / skip. r76 builds that controller on the per-layer store; the multi-hop
question moves from "chain two reads in one pass" to "read → write intermediate →
read across steps", where it belongs.

Multi-hop is therefore parked as an *implicit-mechanism negative* and reopened as a
*memory-management* problem in r76+. Everything else (single-hop recall,
long-context, sparse read, CSA2, update/recency r67) is unaffected.
