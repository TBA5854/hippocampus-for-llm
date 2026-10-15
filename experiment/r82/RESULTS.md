# r82 — route B (loop-gate latent chaining): RESULTS. **Negative.**

## Numbers (best; bpe_hop leak-proof, warm r79 one-store gated)
| arm | recall |
|-----|--------|
| loop-read (loop_iters 2, hard-decode) | 0.2% |
| loop-read broken-chain control | 0.2% |

## Verdict
Looping a read layer — settle → hard-decode the retrieval to a token
(straight-through argmax) → re-query, twice, inject the final hop — does **not**
chain. Chance at 8k, level with the broken control. The hard-commit-per-loop (the
fix for r75's soft-in-forward accumulation) was not enough: the intermediate the
loop decodes is wrong often enough that the second hop lands nowhere.

This is the seventh consecutive negative for **latent** multi-hop (r68 shared, r70/71
per-layer, r72 leak-proof, r73 reindex, r74 e2e-grad, r75 pointer, r82 loop-gate).
The consistent lesson stands: composition does not happen inside the store's
retrieval path, however the intermediate is threaded, because a retrieved
representation is not a reliable next-key.

## Contrast with route A
Route A (explicit token scratchpad, r83) makes the intermediate a real emitted
token, so each hop is a clean single-hop lookup — and its first smoke already shows
both hops as genuine retrievals (hop1 98%, hop2 climbing). Latent chaining (B) is
the hard bet that keeps failing; explicit CoT (A) is the one that works. Route B is
parked; the learned continue/halt gate is not pursued unless A's route is exhausted.
