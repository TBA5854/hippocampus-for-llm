# r86 — memory transplant: stateless weights, stateful store

**Decisive on both halves. (1) The store IS the state: one frozen model answers a query
at full recall only when the matching store is loaded, at chance otherwise — swapping
the store swaps the memory with no weight changed. (2) The state is bound to its weights:
a store written by a different model does not read back (0.0%) even though the facts are
physically present — memory is portable across a store-swap on the SAME weights, not
across different weights.**

## Setup
Two INDEPENDENT models (different seeds → different token embeddings), matched shape so
their stores are shape-compatible: `d256 L6`, `bpe_recall`, 16 facts, `M=512`, plain
store (value-skip-norm, NO write-op gate → both cold-bootstrap per r66), 12k steps.
Own-store recall (the prerequisite for the transplant to be meaningful):
- **modelA (seed 0): 98.8%** — **modelB (seed 1): 100.0%.** Both work on their own store.
Then `mech.persist swap` builds two stores per model and evaluates the full matrix
(in-process transplant, the save/load fingerprint bypassed).

## Result (recall)
| eval | recall | reading |
|------|--------|---------|
| A, store1, query1 (own) | **100.0%** | model A on its own store 1 |
| A, store2, query2 (own) | **96.9%**  | model A on its own store 2 |
| A, store2, query1 (swap) | **0.0%** | wrong store loaded → memory absent |
| A, store1, query2 (swap) | **0.0%** | wrong store loaded |
| A, store1, query1, scrambled | **3.1%** | matched control (chance) |
| B, storeB, queryB (own) | **100.0%** | model B on its own store |
| **B ← A's store1, query1 (foreign)** | **0.0%** | model B reads model A's store, A's facts |
| **A ← B's store, queryB (foreign)** | **0.0%** | model A reads model B's store, B's facts |

## Reading
1. **Same weights, swapped store = swapped memory.** The diagonal (own store) is 97–100%;
   the off-diagonal (the *other* store loaded) is 0.0%, and scrambling is at chance. The
   frozen model's behaviour is set entirely by which store is loaded — the store carries
   the state in full, the weights carry none of it.
2. **The store is model-specific.** Both models read their OWN store at 98.8–100%, but a
   store written by the *other* model reads back at **0.0%** in both directions, despite
   the facts being physically present and the store being shape-compatible. The store's
   keys are the writing model's token embeddings; the reading model's exact-match query
   is computed from *its* embeddings, which (different seed) do not align. So the "bad
   shit" prediction holds exactly: transplanting a store to a foreign model corrupts it
   to chance — memory cannot be moved between models by a file copy.

## Verdict
The stateless-weights / stateful-store thesis is confirmed as a literal, measurable
property. Practically: one shared frozen model can hot-swap per-user stores, each
recovering its own memory exactly (multi-tenant memory); but a store outliving a model
version needs re-encoding (decode the store to its transcript — free, r26/sleep — and
re-write under the new model), not a copy. Numbers folded into paper `transplant.tex`.

(An earlier same-model run on the r85 gated `store_f64` model gave the same diagonal
pattern — own 100%/93.8%, swap 0.0%, scrambled 0.0% — so the store-is-state result holds
for the gated managed store too, not only this plain-store pair.)
