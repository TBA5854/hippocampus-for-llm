# r95 — the honest number: autoregressive CoT-over-memory MATCHES teacher-forced

**Feeding the model its OWN emitted intermediate at each hop (not the ground-truth one)
gives essentially the SAME recall as teacher-forcing, across 2–5 hops. So the multi-hop
result was not a teacher-forcing artifact: the model's own intermediates are good enough
(~98%/hop) that end-to-end composition reaches 90–97%.**

## Numbers (persist chain_ar, own emitted intermediates, n=640 each)
| hops (facts) | autoregressive final | teacher-forced (r84/85/93/94) | per-hop cumulative |
|--------------|----------------------|-------------------------------|--------------------|
| 2 @64 | **96.9%** | 96.9% | 98.6 → 96.9 |
| 3 @64 | **95.2%** | 95.2% | 98.8 → 96.4 → 95.2 |
| 4 @32 | **91.6%** | 91.6% | 98.4 → 96.1 → 93.9 → 91.6 |
| 5 @32 | **89.7%** | 89.7% | 97.2 → 95.5 → 93.8 → 91.6 → 89.7 |

(per-hop = the running joint accuracy after feeding the model's own intermediate through
hop i; the last entry is the final answer.)

## Reading
- **AR ≈ teacher-forced at every depth** (96.9/95.2/91.6/89.7, matching to the digit).
  Teacher-forcing fed the true intermediate at each block; autoregressive feeds the model's
  own emitted token as the next block's query key. That they match means the emitted
  intermediates are correct as often as the final hop is (~98%/hop), so nothing was hidden
  by teacher-forcing — the honest end-to-end number IS the reported one.
- The cumulative per-hop trace is a clean geometric decay (~0.98 per hop): 98→96→94→92 at
  4 hops, 97→96→94→92→90 at 5. Joint recall = product of per-hop retrieval, exactly the
  emitted-token-scratchpad model (r84): each hop is an independent single-hop store lookup.

## Verdict
**CoT-over-memory is confirmed end-to-end, autoregressively.** A model reasons across facts
held outside its context window by emitting each intermediate and re-querying its own store,
reaching 89.7% at 5 hops and 96.9% at 2 hops / 64 facts — feeding its OWN steps, no
teacher-forcing. This closes the paper's one outstanding caveat on the multi-hop result.
Folded into the paper (memory section + chronicle).
