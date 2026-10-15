# r72 — leak-proof multi-hop: does EITHER store genuinely chain?

r70/r71: the single-chain hop task leaked (tail always last-written value, query
always head), so "return most-recent value" scored ~85% blind. r72 removes the leak:
`bpe_hop --n-chains 4` builds 4 independent 2-hop chains, shuffles ALL their facts
into seg0, and queries a RANDOM chain's head. "Return the last value" now yields
some other chain's tail (~chance); only genuine chaining from the queried head
reaches its own tail.

Matched, both stores, warm from each store's r70 1-hop base, seed 0:
- **box1 per-layer multistore**: r72_pl_hop2_lp (8k) + r72_pl_broken_lp (control, 4k).
- **box2 shared** (value-skip + recency): r72_sh_hop2_lp (8k) + r72_sh_broken_lp (4k).

Pass (per store): hop2_lp well above its broken-chain control AND above the
return-last-value chance (~1/(chains*hops)). Broken-chain must fall to chance —
that is what a real chainer does and the single-chain task could not show.

Expected, honestly: both likely fail (r68 shared can't chain; r71 showed per-layer's
apparent success was the leak). If so, multi-hop needs an explicit inter-layer key
transform — a learned reindex that re-lands a retrieval as a valid exact key, i.e.
the thalamic relay — which becomes the r73 build. A genuine above-control result on
either store would instead say that store chains once the shortcut is denied.
Report best AND final.
