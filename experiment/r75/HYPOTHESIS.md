# r75 — the pointer loop: ship a CLEAN decoded key, not the state

Every implicit route failed (r68 shared, r70/71 per-layer leak, r72, r73 reindex,
r74 e2e-grad) because we fed the raw retrieved value backward as the next key, and
that value is off the token-embedding manifold. r75 ships only a **pointer**: decode
the retrieval onto the token-embedding manifold (`p=softmax(beta·cos(r,E)); key=p·E`)
and re-query with that clean, near-token key. Only the key crosses to the query
channel — never the early residual — so exact-match and causal cleanliness hold. The
embedding table is the correct fixed basis the free linear reindex lacked.

`hop_key="pointer"` + `--hop-grad` (train the decode end-to-end), shared store,
leak-proof task (n_chains 4). Both warm from a shared 1-hop base.
- box2 (seed 0, warm r70_sh_recall): r75_ptr_hop2 (10k) + r75_ptr_broken (control, 4k).
- box1 (seed 1, warm r73_sh_recall_s1, 96.4% base): r75_ptr_hop2_s1 + r75_ptr_broken_s1.

Pass: ptr_hop2 >> broken control AND >> chance, at both seeds. That is genuine 2-hop
composition via a decoded-pointer loop — the thalamus-faithful mechanism (a specific
gated query, not a state copy). If still chance, the pointer alone is not enough and
r76 adds intermediate-token aux supervision (teach the decode to emit B directly).
Report best AND final; broken control is the decider.
