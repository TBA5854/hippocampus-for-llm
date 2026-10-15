# r79 — one shared store + gated ops: does the destination match the per-layer scaffold?

r78 proved the managed memory viable on the PER-LAYER store (95.9% recall / 96.2%
update, bounded). But per-layer keeps N copies of every fact (one store per layer) —
the space-inefficient scaffold. The destination is ONE shared store, so content is
stored once (~6× fewer TOTAL slots). r79 tests whether the shared store + the same
gated ops matches the per-layer result — i.e. whether per-layer read ADAPTERS are
even needed.

One shared store (NO --per-layer-store) + --write-op-gate, value-skip + recency (the
r70_sh config), warm from r70_sh_recall, M=512, f16, seed 0.
- box1: r79_shared_gate_recall (bpe_recall, 10k).
- box2: r79_shared_gate_update (bpe_update, 10k).

Reference (r78 per-layer, bounded): recall 95.9%, update 96.2%.

Verdicts:
- shared matches per-layer (recall ~95%, update ~96%, bounded) => one store is
  sufficient, ADAPTERS NOT NEEDED — simplest and best (one copy, per-layer read heads
  already translate via their out-projections). Greenlight multi-hop-as-working-memory.
- shared falls short => the single content does not serve all layers; add a learned
  per-layer read adapter (r80) to translate one content into each layer's space.
Report best AND final + mem occupancy.
