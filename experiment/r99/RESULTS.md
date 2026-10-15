# r99 — multilayer CoT DOES semantics: a parametric hop bridges a synonym to the exact store

**Semantic recall works by REASONING, not by store matching. A CoT whose first hop is
served by PARAMETRIC memory (a fixed synonym→canonical map the model learns in its weights,
never stored) and second by the store (canonical→value) recalls a synonym-queried fact at
90.8% — hop 1 = 100%, hop 2 = 91% — even though the store retrieves synonyms at chance
(r98). The broken-lexicon control (unlearnable map) collapses hop 1 to 2% while the store
hop is untouched (91%), proving hop 1 is a genuinely learned semantic map.**

## Numbers (bpe_semchain, warm r79, 10k; per-hop = char position accuracy)
| run | joint | hop1 syn→canon (parametric) | hop2 canon→val (store) |
|-----|-------|-----------------------------|------------------------|
| main (fixed 64-word lexicon) | **90.8%** | **100%** | 91% |
| broken lexicon (random per example) | **2.0%** | **2%** | 91% |

Robustness (box2): seed 1 reproduces the main (joint **90.6%**, hop1 **100%**), and a smaller
32-word lexicon also learns fully (joint **85.8%**, hop1 **100%**, hop2 86%). Hop 1 = 100%
across seed and lexicon size — the parametric map is reliably learned.

## Reading
- **Hop 1 is learned semantics.** The synonym→canonical map is the SAME across every example
  and is NOT written to the store, so the only way to get hop 1 right is to memorise the map
  in the weights. It reaches 100%. When the map is made unlearnable (random synonym per
  example, the broken control), hop 1 falls to chance (2%) — so the 100% is real parametric
  learning, not a leak.
- **Hop 2 is the store.** canonical→value is a per-example fact in the store; it reads at 91%
  in both the main and broken runs (the control breaks hop 1, not hop 2).
- **Composition.** Joint 90.8% ≈ hop1 × hop2, the same emitted-token-scratchpad chaining as
  r84/r95, now with a *parametric* first hop instead of a stored one.

## Verdict — the resolution of the fuzzy/semantic thread
The store cannot and should not do semantics: it is a geometric nearest-token index
(fuzzy but not semantic, r97/r98; key-enrichment breaks exact match, r92). Semantics lives
in the **weights + CoT**: the model reasons a semantic cue to the exact key, then reads it.
This is the user's reframe ("multilayer CoT should do semantics"), now demonstrated with a
control. On a small PG-19 model the semantic map must be taught (its embeddings don't encode
synonymy, r98); a large LM would supply it for free. The clean division stands: **the store
is a stateless-weights / exact index; meaning is reasoning.** Folded into the paper.
