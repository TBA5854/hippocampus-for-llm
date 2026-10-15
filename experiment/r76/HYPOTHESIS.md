# r76 — gated memory ops on the per-layer store: does the memory manage itself?

New thrust (user reframe): memory is a managed working store, not an append log. A
controller picks an OP per write instead of always appending. v1 is the exact-key
rule (`--write-op-gate`): key already stored -> UPDATE that slot in place (or SKIP if
value identical); key novel -> WRITE-NEW. So the store holds ONE slot per distinct
key = O(facts), and revisions land in place. Per-layer store (all layers see all
facts — simplest place to prove the gate); later r77 collapses to one store + adapters.

Warm from r70_pl_recall (per-layer base, seed 0). Metrics: recall AND mean store
occupancy (`mem N/M`, added to the log).

- **box1 — bounded recall.** bpe_recall, per-layer, M=512, f16: `gate` vs `nogate`.
  Prediction: gate keeps `mem` ~O(distinct keys) (small, bounded) with recall = nogate;
  nogate fills `mem` toward M with one slot per token written.
- **box2 — updates land in place.** bpe_update, per-layer, M=512, f16: `gate` vs
  `nogate`. Prediction: gate overwrites the fact's slot -> latest-value recall high AND
  `mem` bounded; nogate appends a second slot (r67 keep-both, no recency here) -> lower
  latest recall and larger `mem`.

Pass: gate holds recall while `mem` stays far below nogate's; gate lands updates in
place. That is a self-managing, O(facts) memory — the space thesis and the
write/merge/update control unified. Merge (blend agreeing values) and the LEARNED
gate are the next step (r77). Report best AND final, and the mem occupancy.
