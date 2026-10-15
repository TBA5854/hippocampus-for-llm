# r42 — reusing DG for the ambiguity: no-ops on identical keys, collapses on contextual ones

The user's idea: r36 showed the model can't tell the 8 `:` slots apart, and DG (dentate gyrus) is the hippocampal
pattern-separation circuit — so reuse it. This tests both the literal version and the version that could actually
work. 8 facts, seed 1.

| run | exact | char | pos0 | pos1 | gate |
|---|---|---|---|---|---|
| `r36_lang_s1` — layer-0 key, DG off (baseline) | 10.3% | 45.9% | 11.4% | 80.5% | 0.206 |
| `r42_dg_l0` — layer-0 key, DG off (repro) | 10.3% | 45.9% | 11.4% | 80.5% | 0.206 |
| `r42_dg_l0on` — layer-0 key, **DG ON** | 11.7% | 49.9% | **12.5%** | 87.3% | 0.192 |
| `r42_ctx_dg` — **contextual key (L1) + DG ON** | **0.2%** | 3.6% | 4.2% | 3.0% | 0.092 |

## Verdicts

- **DG on the current key no-ops. Confirmed.** `pos0` goes 11.4% -> 12.5% — still exactly 1/n_facts, no separation.
  The reason is structural: with `key_layer=0, key_pos_free`, the key is the raw token embedding of `:`, so the 8
  colon-slots have **byte-identical** keys. DG is a deterministic function; it maps identical inputs to identical
  codes. There is nothing to separate. Pattern separation needs *similar-but-distinct* inputs, and these are not
  distinct at all.
- **The contextual key needed to make DG bite collapses to chance. Confirmed r4.** Taking the key from layer 1
  (so the `:` carries which fact precedes it) drops the whole task to **0.2%** — every metric at chance. This is r4's
  contextual-key drift and r31d's finding from the other side: the exact-token match is load-bearing, and any key
  that is a *contextualised* state rather than the raw token embedding destroys it. The very move that would give DG
  something to separate is the move that breaks retrieval.

## Diagnosis — the idea is sound, the architecture forbids it

Pattern separation is the right tool for confusable keys, and the instinct was correct. But this store's keys are
not confusable-similar — for the shared delimiter they are *identical*, and to make them non-identical you have to
contextualise the key, which this architecture cannot do without losing exact matching (r4, r31d, r42_ctx here).
So DG has no regime to help in here: where keys are distinct (MQAR) it only adds noise (why it was bypassed from the
start), and where they are identical (the delimiter) it cannot manufacture a distinction.

The escapes that work remain the ones that avoid the identical-key situation rather than trying to separate it:
**shift-3** keys on the unique key-letters instead of `:` (r37), and the **curriculum** teaches disambiguation from
a solvable start (r38/r40/r43). Both leave the exact-match property intact.

## What this does NOT show

- One contextual depth (layer 1) and one DG sparsity. A *learned, exactness-preserving* key compressor is a
  different project (OBJECTIVE.md's contextual-key direction, unbuilt) and is not refuted by this — only the
  drop-in-DG version is.
- One seed.
