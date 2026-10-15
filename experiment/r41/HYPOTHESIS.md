# r41 — placement: does the model want to read at more than one layer?

The user's long-standing ask ("r start w end vs r/w at evry layer, both are gated"). This is the free, config-only
rung: reads at every layer against one shared store, write still at the last block. `--read-layers 0 1 2 3 4 5` adds
a gated read head at each block; the model already calls read() at every layer and it no-ops without a head.

Run at 4 facts (which works at 90.3%), so any change is visible against a high baseline rather than at chance.

- **H41a — most gates stay shut.** The project rests on a layer-0 key being near-exact token lookup (r31d: a
  contextualised key loses the exactness). Deep reads query contextualised states, so predicted: the layer-0 gate
  carries the load and deep gates stay near zero. `gate_magnitudes()` reports per layer, so this reads directly off
  the run. Falsified if deep gates open AND recall/char improves over read-at-0-only, which would say retrieval is
  useful at depth and the single early read leaves something on the table.
- Total slots held constant (one shared store, M=512), so this is placement, not capacity — the r4 fake-axis rule.

Writes-at-every-layer needs plumbing (write_layer is one int; forces a shared-vs-per-layer-store choice) and is
deferred to a later round.
