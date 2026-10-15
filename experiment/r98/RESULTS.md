# r98 — the store's fuzzy recall is geometric, NOT semantic, and can't be widened by softening

**Two probes close the store-side fuzzy question. (a) Semantic: querying with a synonym
retrieves at CHANCE (~4.2%) across every model, because the token embeddings barely
separate synonyms from random words (cos 0.91 vs 0.87). (b) Beta sweep: the fuzzy
(approximate-key) basin comes from SHARP nearest-slot selection — softening the read
COLLAPSES it (nn-1 0.6% at β=8) and sharpening saturates it (~75% at β≥64). So the store
does geometric nearest-token recall, not meaning, and it isn't tunable into semantics.**

## (a) Semantic probe — synonym query vs exact vs random (n=24 single-token synonym pairs)
| model | store | exact | **synonym** | random | cos(syn) vs cos(rnd) |
|-------|-------|-------|-------------|--------|----------------------|
| d512/2L | plain | 54.2% | **4.2%** | 0.0% | 0.754 vs 0.682 |
| d384/3L | plain | 50.0% | **8.3%** | 0.0% | 0.827 vs 0.785 |
| d256/6L | plain | 41.7% | **4.2%** | 0.0% | 0.906 vs 0.870 |
| d192/12L | plain | 41.7% | **4.2%** | 0.0% | 0.891 vs 0.839 |
| d256/6L (r92) | plain | 41.7% | **4.2%** | 0.0% | 0.906 vs 0.870 |
| d256/6L (store\_f64) | gated | 50.0% | **0.0%** | 0.0% | 0.909 vs 0.879 |

Synonym recall = chance (4.2%) everywhere. The cause is in the last column: a synonym is
only marginally closer than a random word in embedding space (e.g. 0.906 vs 0.870), so the
argmax-cosine read cannot use meaning. The r97 fuzzy recall was purely geometric (nearest
token), not semantic.

## (b) Settle-sharpness sweep — d512/2L, nn-rank recall vs read β
| β | exact (nn0) | nn1 | nn2 | nn3 |
|---|-------------|-----|-----|-----|
| 8 (soft)   | 96.9% | **0.6%** | 0.2 | 0.2 |
| 16         | 98.1% | 33.1% | 20.0 | 15.0 |
| 32 (default)| 98.5% | 70.6% | 57.9 | 51.0 |
| 64         | 98.8% | 74.4% | 62.9 | 55.2 |
| 128 (sharp)| 99.0% | **74.8%** | 62.5 | 54.8 |

The fuzzy basin is widest when the read is SHARP: argmax commits to the single nearest slot
(the right one for a near query), while a soft read averages many slots into a wrong blend.
So the approximate-key basin cannot be widened by softening — it is fixed by embedding
geometry and saturates by β≈64.

## Verdict
Store-side semantic recall is a dead end within the exact-match design: the read is
geometric (nearest token), not semantic; enriching the key breaks exact match (r92); and
softening the read only destroys the geometric basin (r98b). **Semantics must come from
REASONING, not store matching** — the model chaining a semantic cue to the exact key via
CoT (the r99 direction, per the user: "multilayer CoT should do semantics"). The store
stays a clean exact/approximate index; the weights + CoT supply meaning.
