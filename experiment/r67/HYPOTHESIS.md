# r67 — update / recency: can a re-stated fact return its NEWEST value?

Today a re-stated fact writes a NEW ring slot, so the store holds two slots with
the same key token and different values; `settle` blends/argmaxes them — no
"latest". r67 keeps both slots and adds a **recency read**: bias the match toward
the slot written last (global `slot_seq`), so the newest value wins while the old
one stays queryable. Shared store, ml placement (read+write @ all 6 layers), warm
from r66_mlcurr_f16 (100% @16 facts). Task `bpe_update`: fact stated once in seg0,
re-stated once in a later seg, query returns the LATEST value.

Arms (box1, all warm from r66_mlcurr_f16.pt, n_facts 16, seed 0):
- **recency_off** — bpe_update, no recency (baseline). Because the flagship key is
  position-FREE, there is no positional recency in the key; this arm tests the
  user's "learnt token positions act as recency" hypothesis directly. Prediction:
  latest-recall stalls near chance / blend (the model cannot tell which slot is
  newer), proving the explicit bias is needed.
- **recency_on** — bpe_update + `--recency-read`. Prediction: latest-recall ~100%,
  and high even at step 0 (the bias is mechanical, like the sparse/pool read).
- **nonupd** — recency_on + `--query-nonupdated` (query a fact that was NOT
  updated). Control: recency must NOT disturb ordinary recall (stays ~100%).

Pass: recency_on ≫ recency_off on latest-recall; nonupd ~100%. If recency_off is
already high, position-in-key gives recency for free (user right); if only
recency_on is high, the explicit slot_seq bias is required (mechanism needed).
Report best AND final.
