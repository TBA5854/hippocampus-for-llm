# r5 — results: layer-0 keys make 8-fact recall 100% on every seed

## Numbers (8 facts, 4000 steps; value from the last block, shift 0, identity query init)

| run | key + read | beta_grad | extra | seed | best | final |
|---|---|---|---|---|---|---|
| r3_f8 (ref) | layer 1 | 4 | — | 0 | 83.9% | 83.9% |
| r4_f8_s1 (ref) | layer 1 | 4 | — | 1 | 6.1% | 3.9% |
| r4_f8_s2 (ref) | layer 1 | 4 | — | 2 | 4.1% | 3.6% |
| **r5_k0** | **layer 0** | 4 | — | 0 | 100.0% | **100.0%** |
| **r5_k0_s1** | **layer 0** | 4 | — | **1** | 100.0% | **100.0%** |
| **r6_f8_s2** (from r6) | **layer 0** | 4 | — | **2** | 100.0% | **100.0%** |
| r5_floor_s1 | layer 1 | 4 | grad floor 0.1 | 1 | 5.5% | 3.9% |
| r5_bg16 | layer 1 | 16 | — | 0 | 66.4% | 64.7% |
| r5_bg16_k0 | layer 0 | 16 | — | 0 | 100.0% | 100.0% |

```
k0      s0   3  2  3  5 99 100 100 ...        lift-off by step 1250
k0      s1   3  4  8 99 100 100 ...           lift-off by step 1000
k0      s2   2  3  3  9 100 100 ...           lift-off by step 1250
floor   s1   3  3  3  2  2  2  2  3  3  3  3  5  3  3  4  4
bg16    s0   3  2  3  2  4  3  5  6  5 21 47 57 63 66 65 65
```

Correct-slot reads for every layer-0 run: **100% from the first eval (step 250) to the last**, weight mass
1.00 on the correct slot, never decaying.

## Verdicts

- **H5b (identity keys beat 83.9%) — confirmed.** 100.0% vs 83.9% on the same seed.
- **H5d (layer-0 keys survive the failing seed) — confirmed**, and seed 2 (run in r6) as well: **3 of 3
  seeds at 100%**, where layer-1 keys got 1 of 3. Lift-off is also earlier (step 1000-1250 vs ~1750).
- **H5e (a gradient floor survives it) — falsified.** 3.9%, indistinguishable from no floor. Alignment still
  fell 37% -> 2% by step 750. Extra gradient to the query cannot hold alignment when it is the *keys* that
  move.
- **H5a (sharper surrogate beats 83.9%) — falsified as stated, with a useful side result.** 64.7% final, below
  r3_f8. But its alignment *rose* during training (21% -> 70%), which never happened at `beta_grad 4`
  (alignment ended where init put it). A steeper surrogate lets matching learn; it just lifts off later
  (step ~2250). Worth remembering for contextual keys, where matching must be learned rather than given.
- **H5c (they compose) — untestable here.** `bg16_k0` is 100%, but `k0` alone already is; there is no headroom
  left to show composition.

## Diagnosis

The seed failure was the drift race r4 inferred. The evidence is now behavioural and one-variable:

- Same seed (1), same everything, keys moved from layer 1 to layer 0: **3.9% -> 100%.**
- Same seed (1), layer-1 keys, extra read-path gradient: **3.9% -> 3.9%.**

Moving the key to a depth with no trained context mixing removes the failure; helping the query adapt
does not. The keys were moving under the query, not the query failing to learn.

## Caveat that bounds the claim

A layer-0 key is token embedding + position embedding, so matching here is close to **exact token
lookup**. That is a legitimate and complete test of what this round asked — does a store written at the
last block and read at the first persist and deliver content inside the forward pass — but it is **not**
evidence the store can retrieve by meaning. A paraphrased or contextually-cued query would not match.
Getting robust *contextual* keys (layer >= 1 without the drift race) is open; r5_bg16 suggests a steeper
surrogate is part of that answer.

## Objective status after r5

| requirement | status |
|---|---|
| read at the first layer, write at the last block | **done** (read and key at block 0, value from block 5) |
| >= 8 facts out of context | **done — 100.0% on 3 of 3 seeds** |
| controls at chance | HC off 2.5% (r4); scrambled for the layer-0 key space running in r6 |
| store persisted + reloaded | **done** on r3's checkpoint (81.25% == 81.25%); to repeat on a layer-0 checkpoint |
