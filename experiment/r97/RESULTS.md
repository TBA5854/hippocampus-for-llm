# r97 — fuzzy recall is partially EMERGENT in the plain store, arch-dependent, and KILLED by the gate

**The "exact" layer-0 match is really argmax-cosine, so a query token that is only a
near-embedding neighbour of a stored key still retrieves its value — 33–73% at the nearest
neighbour (~20× chance), no training. It is strongest in wide/shallow models and
ELIMINATED by the managed (gated) store. So the plain store is already partly
hippocampus-like (soft pattern completion); the managed store is a pure dictionary.**

## Near-neighbour recall by model (persist fuzzy, n=640; chance ≈ 1/facts)
| model | store | exact (nn0) | nn1 (cos) | nn2 | nn5 | nn10 |
|-------|-------|-------------|-----------|-----|-----|------|
| d512 / 2L (wide)  | plain | 98.8% | **73.0%** (0.83) | 58.8 | 46.6 | 32.7 |
| d384 / 3L         | plain | 97.5% | **62.7%** (0.89) | 48.6 | 35.3 | 27.2 |
| d192 / 12L (deep) | plain | 96.9% | **61.9%** (0.95) | 48.0 | 37.5 | 25.6 |
| d256 / 6L         | plain | 84.7% | **33.0%** (0.94) | 24.7 | 16.4 | 12.8 |
| d256 / 6L (r92 recall) | plain | 84.7% | 33.0% | 24.7 | 22.7 | 16.4 |
| **d256 / 6L (store\_f64)** | **gated** | 95.9% | **0.2%** | 0.2 | 0.0 | 0.0 |

## Reading
1. **Emergent, and real** — a token never shown, only near a stored key, retrieves its
   value 33–73% at rank 1 (chance ~2–3%). The store is a soft nearest-neighbour lookup with
   a basin around each key, not a hard hash. Recall decays smoothly with embedding distance.
2. **Arch-dependent** — the wide/shallow model has the widest basin (d512/2L 73% even though
   its neighbours are FARther, cos 0.83), while the weak d256/6L (lowest exact, 84.7%) has
   the narrowest (33%). Wider embedding space ⇒ more spread keys ⇒ wider fuzzy basin.
3. **The gate kills it** — the managed store (write-op gate + recency, store\_f64) drops to
   **0.2%** at rank 1: deduping to one exact slot per key token and biasing to the newest
   slot turns the soft lookup into a hard exact table. This is a real **trade-off**: the
   bounded, updatable, O(facts) managed store sacrifices the plain store's emergent fuzzy
   tolerance.

## Caveats
- "Nearest embedding neighbour" is a surface-embedding neighbour, not a verified synonym —
  this shows *approximate-key* recall, not yet *semantic* recall (needs a meaning-pair source).
- Different models have different neighbour cosines, so cross-model numbers aren't a
  perfectly controlled comparison; the within-model rank decay and the plain-vs-gated gap are.

## Verdict
Answers "is fuzzy already there": **yes, partially, in the plain store** — the exact-match
design is a soft NN lookup, giving free approximate recall (widest in wide models), but the
managed store removes it. Open next: (a) is it semantic (meaning-pairs vs cos-matched
random)? (b) can a softer read widen the basin without dissolving exact discrimination
(beta sweep)? (c) can the managed store keep some fuzz? Folded into the paper limitations.
