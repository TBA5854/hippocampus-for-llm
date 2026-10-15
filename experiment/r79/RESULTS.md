# r79 — one shared store + gated ops: RESULTS. **One store suffices. Adapters not needed.**

## Numbers (best; one shared store, --write-op-gate, warm r70_sh_recall, 10k)
| task | recall | mem (slots/512) | per-layer ref (r78) |
|------|--------|-----------------|---------------------|
| bpe_recall (shared+gate) | **96.2%** | 198 | 95.9% |
| bpe_update (shared+gate) | **96.6%** | 251 | 96.2% |

## Verdict
One shared store with the gated ops **matches the per-layer scaffold** (96.2/96.6 vs
95.9/96.2) at the same per-store occupancy — but it is ONE store, not six, so **~6×
fewer TOTAL slots** (198 vs ~1188 across six per-layer stores). Recall and update
both hold.

So the space-efficient destination is reached, and it is the **simplest** form:
- content is stored **once**;
- the existing **per-layer read heads** (each with its own out-projection / gate /
  alpha) already translate that single content into each layer — so a separate learned
  per-layer **adapter is unnecessary** in this regime;
- the gated ops keep it bounded (O(distinct keys)) and land updates in place.

The managed, bounded, single shared store is complete: it holds one slot per fact,
recalls at ~96%, revises in place, and (from earlier rounds) holds ≫ the context
window with sparse O(1) reads and 4-bit storage.

## What this leaves
The one open frontier is **composition / multi-hop**, which the implicit attempts
(r68–r75) showed is the wrong organ for a single-pass lookup. On this proven single
store it reopens in the working-memory frame: **write the computed intermediate, read
it back across steps** — i.e. explicit intermediate reasoning, where each hop is a
clean single-hop lookup (which works at 96%). r80 tests that directly.
