# r89 — DG pattern separation: does a fine (bigram) key resolve key-token collisions?

## Problem
The exact single-token key cannot separate facts that SHARE a key token. r36 measured
this as the 1/n draw: with C facts keyed by the same token, a query lands on 1/C of them.
Recency (r67) picks the newest (right for updates, wrong for distinct facts); the write-op
gate MERGES same-key slots (worse). This is the collision axis the exact-match design has
been dodging, and the honest limit behind "more than a context window" (keys collide
constantly at scale).

## Mechanism (dentate gyrus, done compatibly)
DG separates similar-but-different inputs; here the difference is the CONTEXT token. Lay
each fact out as `c k v` (context, shared key, value). Turn the position-free layer-0 key
into a BIGRAM: `emb(k) + alpha*emb(c)` (`--fine-key`, alpha=`--fine-key-alpha`) — a coarse
token bucket PLUS a fine context term. The query `<q> c k` has the same (c,k) adjacency,
so it reproduces the composite exactly: exact-match is preserved (unlike r31/r42, which
contextualised the WHOLE key and collapsed), but facts sharing k are pulled apart by c.
This directly addresses r36's "a single-token key cannot address a multi-token key," and
the two-component code is the "2xfp8" idea (coarse + fine).

## Predictions
- **Exact-only baseline** (`--collisions-per-key C`, no fine key): recall ~ **1/C**
  (C=2 -> 50%, C=4 -> 25%, C=8 -> 12.5%) — the collision draw.
- **Fine key on**: recall -> **high** (~single-fact recall) at every C — the bigram
  separates the colliding facts while staying exact.
- **Scrambled-context control** (`--scramble-context`): query with the WRONG context ->
  back to ~1/C or worse (proves the lift is the context term, not leakage).

## Runs (warm from r79; plain store, no gate; 6k steps; box param via qrun)
C=4: exact / fine / fine+scramble ; C=2: exact / fine ; C=8: exact / fine.
Pass = fine >> exact at each C, exact tracks 1/C, scramble control at chance. A win is
DG pattern separation working on a real collision axis — the step from dictionary
(one slot per token) toward hippocampus (context-separated memories), and it feeds the
still-open fuzzy/semantic-recall thread (relax the coarse term next).
