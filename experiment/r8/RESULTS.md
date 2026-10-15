# r8 — results: position was not the cause; the learned query steers away from facts

## What ran

| run | keys | best | final | correct-slot reads |
|---|---|---|---|---|
| r8_w0s1 | layer 0, **position-free**, learned query (identity init) | 4.7% | stopped at step ~1750, chance | 100% at 250 and 500 -> **0.0% from step 750** |
| r8_w5s1, r8_w5s0, r8_w3s1 | same key | — | — | stopped before running |

## Verdicts

- **H8e (position-free matching holds all run) — falsified.** Removing position made the collapse come
  *earlier* (step 750 vs r7's 1250), not go away. So position was at most a contributor; r7's diagnosis
  named where the reads went, not why the query moved there.
- H8a-d — not tested; carried to r9.

## Diagnosis — measured on r7_w0s1's checkpoint (same collapse, checkpoint available)

Two candidate causes were checked directly rather than assumed:

**1. Centring scale mismatch — ruled out.** The query is normalised before the running key mean is
subtracted while keys are raw, which would swamp identity if that mean outgrew unit norm. Measured:
`|key_mean|` 0.48, `|stored key|` 0.90, `|tok_emb|` 0.70 — all below 1. And top-1 correct stays **0.000**
with centring as coded, with no centring, and with both sides normalised before centring.

**2. The learned query projection — confirmed.**

| checkpoint | `|W - I| / |I|` | learned query | untrained identity query | token-only query |
|---|---|---|---|---|
| r5_k0 (tiled, works) | 0.186 | 1.000 | 1.000 | 1.000 |
| **r7_w0s1 (single occurrence, collapsed)** | **0.170** | **0.000** | **0.938** | **0.938** |

The collapsed model's query projection is no further from identity than the working one's, yet that small
learned deviation takes matching from 94% to 0%. Random drift of that size does not zero a match; this is a
targeted change the gradient made.

Plausible reason, left there deliberately (mech-interp deferred): **one query projection serves every
position.** With single-occurrence facts, segment 0 is mostly text, so each example sends gradient into the
projection from ~380 text positions and one answer position (weight 5). What helps or harms reads at text
positions decides where the projection goes, and the answer position loses. Tiled training never had this
conflict: segment 0 held no text.

## Fix and next round

With position-free layer-0 keys, matching is exact token lookup, so a learned query adds nothing except a way
to break it. `--freeze-query` holds the projection at identity (tested: unchanged through an optimiser step;
smoke run 100% matching). r9 reruns the four value sources with it.
