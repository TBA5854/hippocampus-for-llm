# r78 — confirm the ceiling: RESULTS. **Bounded gate reaches FULL recall. Zero cost.**

## Numbers (continued from r77 to 12k total; per-layer store, --write-op-gate)
| task | recall @12k | mem (slots/512) | no-gate reference |
|------|-------------|-----------------|-------------------|
| bpe_recall (gate) | **95.9%** | **198** | 94.1% at mem 508 (full) |
| bpe_update (gate) | **96.2%** | **251** | 7.8% at mem 508 (full) |

## Verdict
The 86→96 climb from 6k→12k confirms the r77 diagnosis exactly: the recall gap was
**under-training**, not a cap and not prose-clobber. At convergence the bounded gate
**matches/beats the unbounded no-gate recall (95.9 ≥ 94.1) at 2.6× fewer slots**, and
lands updates at 96.2% (vs 7.8% for append-both).

So the managed memory is **fully viable with zero recall cost**:
- **Bounded** — 198–251 slots vs 508, O(distinct keys).
- **Recall-preserving** — 95.9%, on par with the redundant store.
- **Updates land** — 96.2% in place.

The crude exact-key gate needs no salience refinement in this regime. The controller
concept (write-new / update / skip by key) is proven.

## Greenlight
The user's checkpoint ("if viable for memory, good for the next one store") is met
cleanly. Next major step: **one shared store + per-layer read** (the space-efficient
destination — content stored once instead of once-per-layer, ~6× fewer TOTAL slots
than the per-layer scaffold). r79 tests whether one shared store + the gated ops
matches this per-layer result; per-layer adapters are added only if the single
content does not serve all layers.
