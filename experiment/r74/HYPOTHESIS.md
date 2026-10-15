# r74 — end-to-end gradient through hops: the last lever for multi-hop

r73: the reindex failed because hop_q is detached and only the current hop trains,
so no signal reaches the intermediate key. r74 removes the detach (`--hop-grad`):
the final-answer loss backpropagates through EVERY hop into the read path (and the
reindex), training the chain jointly. Shared store, leak-proof task (n_chains 4),
hop_key retrieved.

- **box2**: hop_grad + **reindex** (the full fix) — hop2 (10k) + broken control (4k),
  warm from r70_sh_recall.
- **box1**: hop_grad, **no reindex** (does end-to-end gradient alone suffice?) —
  hop2 (10k) + broken control (4k), warm from r73_sh_recall_s1 (seed 1, 96.4% base).

Pass: hop2 >> broken control AND >> chance (on either arm). That would finally be
genuine 2-hop composition on the shared store, trained end-to-end. If BOTH stay at
chance, multi-hop is a hard limit of this store design as configured — the retrieved
value does not carry enough recoverable token identity even with joint training —
and the honest conclusion is that composition needs a different value representation
(e.g. store the token embedding as the value) or an explicit two-read decoder, which
becomes a design change rather than a knob. Report best AND final; broken control is
the decider.
