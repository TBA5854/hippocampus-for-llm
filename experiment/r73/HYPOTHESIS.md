# r73 — learned REINDEX: does re-landing a retrieval on the key manifold unlock chaining?

r68/r72: the shared store can't chain because a raw retrieved value
(scale·(norm(last)+norm(embed))) is off the token-embedding key manifold, so the
next hop's exact match fails. r73 adds a LEARNED linear `reindex` applied to the
retrieval before it becomes the next hop's key (init at identity; trainable via the
current hop's read even though hop_q is detached). This is DeepSeek's "reindex" /
the user's thalamic relay. Shared store, leak-proof task (`--n-chains 4`), hop_key
retrieved, value-skip + recency (same as r70 shared base).

- **box2 (primary, seed 0)**: r73_sh_reindex_hop2 (warm r70_sh_recall, 10k) +
  r73_sh_reindex_broken (broken-chain control, 4k).
- **box1 (replication, seed 1)**: r73_sh_recall_s1 (shared base, 8k) ->
  r73_sh_reindex_hop2_s1 (8k).

Pass: reindex hop2 >> its broken-chain control AND >> the return-last-value chance,
replicated at seed 1. That would mean the learned reindex unlocks genuine 2-hop
composition on the shared store. If it stays at chance, the retrieval carries too
little of the value token's identity for any linear map to recover, and multi-hop
needs end-to-end gradient through hops or a different value representation. Report
best AND final; the broken-chain control is the decider (r70's lesson).
