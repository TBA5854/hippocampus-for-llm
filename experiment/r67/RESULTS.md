# r67 — update / recency: RESULTS

**Verdict: a re-stated fact can return its NEWEST value, and the recency read is
what makes it work early and well. With a position-free key the model does NOT get
recency for free — it must be given the explicit write-order bias.** Positive.

## Numbers (latest-value recall, best = final)

| arm | step 500 | final (6k / 4k) |
|-----|----------|------|
| recency_off (baseline) | 42.2% | **88.9%** |
| recency_on (`--recency-read`) | **82.3%** | **96.7%** |
| nonupd control (recency_on, query a NON-updated fact) | — | **97.5%** |

All warm from r66_mlcurr_f16 (ml placement), `bpe_update`, n_facts 16, seed 0.

## Reading it
- **Recency helps, immediately and at the ceiling.** With the explicit slot_seq
  bias the model is already at 82.3% at step 500 (vs 42.2% without) and finishes at
  96.7% (vs 88.9%). The bias gives "latest wins" for free at read time; without it
  the model can *learn* to prefer the newer slot, but slower and to a lower ceiling.
- **Position is NOT recency here (answers the hypothesis).** The flagship key is
  position-free, so there is no positional order in the key. recency_off starting
  at ~chance-between-the-two-values (42%) and only climbing with training confirms
  the model gets no free recency from position — the explicit global write-order
  (`slot_seq` / `norm_age()`) is what supplies it. Had position-in-key carried
  recency, recency_off would have started high; it did not.
- **Recency does not disturb ordinary recall.** Querying a fact that was never
  updated stays at 97.5% with recency on — the bias only breaks ties among slots
  that share a key token, so single-slot facts are untouched.
- **Both slots stay stored.** The old value is not overwritten; recency just
  ranks the newer one first, so history remains queryable (the design the user
  chose over destructive overwrite).

## Caveat
recency_off reaching 88.9% means the readout/gate can partly learn latest-selection
on its own given the update-task training; the clean "mechanism vs learned" split
would be a step-0 (no-train) probe. The matched-training comparison still isolates
the recency read's contribution (+7.8 pts final, +40 pts at step 500).

Single seed. `bpe_update` states each fact once (no tiling) so global write order
cleanly separates the stale slot from the update.
