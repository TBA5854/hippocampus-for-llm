# r71 — harden the r70 multistore multi-hop result

r70: per-layer multistore reached 85.0% 2-hop vs shared 8.9%. Three checks before
we trust it:

box1 (per-layer, warm from r70_pl_recall.pt, seed 0):
- **pl_broken** — `bpe_hop --n-hops 2 --broken-chain`: cut the last link so the head
  cannot reach the tail. If the 85% was real chaining, this must fall to chance. If
  it stays high, the model was shortcutting, not chaining (the result would be void).
- **pl_hop3** — `bpe_hop --n-hops 3`: does it chain A->B->C->D, or run out of depth?

box2 (per-layer, seed 1, full curriculum):
- **pl_recall_s1** (8k) -> **pl_hop2_s1** (6k): replicate the 85% at a second seed.

Pass: pl_broken << 85% (ideally ~chance), pl_hop2_s1 ~ 85% (replicates), pl_hop3
tells us the depth limit. If pl_broken stays high, the r70 headline is not chaining
and must be revised. Report best AND final.
