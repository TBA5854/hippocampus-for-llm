# r74 — end-to-end gradient through hops: RESULTS. **Negative — exhausts the implicit approaches.**

## Numbers (best AND final; shared store, leak-proof task)
| arm | hop2 | broken control |
|-----|------|----------------|
| grad + reindex (seed 0) | 0.5% (final 0.3) | 0.3% (final 0.0) |
| grad only, no reindex (seed 1, base 96.4%) | 0.2% (final 0.0) | 0.2% (final 0.0) |

## Verdict
Training the chain end-to-end (no detach on the retrieval fed to the next hop) does
nothing — chance at both seeds, with and without the reindex, level with the
broken-chain controls. Combined with r68/r72/r73, **every implicit route to
multi-hop has now failed**: shared read, per-layer stores, a learned linear reindex,
and end-to-end gradient. Feeding the raw retrieved value backward as the next key
cannot be made to work, because that value (`norm(last)+norm(embed)`) is off the
token-embedding key manifold and joint training does not discover a correction.

## The one path left, and why it's different (r75)
The failure is that we ship the whole messy retrieved *state* backward. The fix
(user's design, thalamus-faithful) is to ship only a **clean decoded key** — a
*pointer* — and re-query:
- **Project the retrieval onto the token-embedding manifold** (decode it through the
  embedding table: `p = softmax(β·norm(r)·norm(E)ᵀ); key = p·E`), so the next hop's
  query is ≈ an actual token embedding, discarding the off-manifold noise the reindex
  could not remove. The embedding table is the correct fixed basis a free linear map
  lacked.
- **Only the key crosses back**, into the query channel — never the early residual —
  so the exact-match property and causal cleanliness are preserved.
- Trainable end-to-end (the decode is a differentiable softmax); r76 can add an
  auxiliary loss on the intermediate token if the pointer alone is not enough.

r75 implements this as `hop_key="pointer"`. If it too is at chance, multi-hop needs
an explicit two-read decoder with intermediate-token supervision (r76), i.e. a
training-signal change, not just an architecture knob.
