# r70 — does a per-layer MULTISTORE enable multi-hop where the shared store failed?

r68/r69: with ONE shared store, every read layer queries the same token key and
retrieves the same thing, so chaining fails (retrieved value is not a clean exact
key). 1-hop works (r69: 98.8%); 2-hop does not. Idea (user): give each layer its
OWN store. Layer 0 is token-keyed (resolves A->B), deeper layers are keyed on the
evolving residual (which now carries B), so layer 1 reads its own store and gets C
-- "lower layers do one hop, higher layers the next", with distinct stores so the
A->B and B->C slots never collide on one key.

Matched A/B, same curriculum (recall -> hop2), same hop task, seed 0, n_facts 16,
ml placement (read+write @ all 6 layers):
- **box1 MULTISTORE** (`--per-layer-store`): stageA per-layer bpe_recall (8k, base
  1-hop) -> stageB per-layer bpe_hop n_hops 2 (6k, warm from stageA).
- **box2 SHARED control** (value-skip + recency, the proven single-store config):
  stageA shared bpe_recall (8k) -> stageB shared bpe_hop n_hops 2 (6k). Reproduces
  the r68 failure under the identical curriculum, isolating per_layer as the only
  difference.

Pass: box1 multistore hop2 recall >> box2 shared hop2 (and >> chance). Then the
per-layer multistore chains and the shared store needs the reindex/relay. If box1
ALSO fails, multi-hop needs an explicit inter-layer key transform regardless of how
many stores -- pointing squarely at the learned reindex / thalamic relay. StageA
recall confirms both bases learn 1-hop before we judge 2-hop. Report best AND final.
