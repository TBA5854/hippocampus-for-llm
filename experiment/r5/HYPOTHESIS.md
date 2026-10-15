# r5 — break key-vs-key confusion

## Ideate (from r3 + r4 offline)

r3's 8-fact model reaches 83.9%. The offline miss taxonomy on its checkpoint (`r4/offline/tax_r3_f8.json`)
is unambiguous: **every matching miss lands on another fact's key** — 17.5% of queries at 8 facts, 10% at
2 — and **none** on distractor text, value slots or the query marker. Content is solved (0.3% right slot
+ wrong answer). And r3 showed alignment ends where identity init put it (~84-90%): training does not
improve matching.

So the store confuses one key symbol for another, and the query projection does not learn to separate
them. Two candidate reasons, each with a one-variable test:

1. **The surrogate gradient is too flat to separate near-tied keys.** Backward uses `beta_grad=4`. Two
   keys whose cosines to the query differ by ~0.1 get almost equal soft weight at beta 4, so the gradient
   that would push the right one up and the wrong one down is tiny. `beta_grad=4` was chosen in v0.3 to
   stabilise a *random* query init (the 67.2% -> 26.7% decay); with an identity start that constraint may
   no longer bind.
2. **Layer-1 keys are not identity-dominated enough.** A layer-1 state has been through block 0's
   attention, so a key `k_j` in the tiled fact segment carries its context (`... v_{j-1} k_j`). Different
   key symbols share that context structure, and centring subtracts a mean dominated by distractor text
   (254 of 384 slots), leaving a common "key symbol" offset in every key. Reading and keying at **layer 0**
   (token + position embedding, before any context mixing) removes the shared context.

## Hypotheses

- **H5a — sharper surrogate.** `beta_grad 16` lifts 8-fact recall above r3's 83.9%, and does so by moving
  the `other_fact_key` share of misses down.
- **H5b — identity keys.** `key_layer 0, read_layers 0` lifts 8-fact recall above 83.9%.
- **H5c — they compose.** Both together beat either alone.

## Falsifiers

- `bg16` destabilises (peak then decay, the v0.3 failure) -> the random-init constraint still binds.
- `k0` is worse -> position embedding in the key adds more confusion than context removal saves; the next
  step would be a position-free key, which risks turning matching into an exact token lookup and must be
  reported as such.
- Neither moves -> confusion is not in the key representation or the surrogate; next candidate is the
  query projection's capacity to learn at all (learning rate on the read head, or the gate).

## Added before launch: r4_f8_s1 collapsed

r3's config on seed 1: identity-slot alignment 30% (step 250) -> 11% -> 7% -> **0% at step 1000**, with
the gate never opening (0.005). So 83.9% is seed-dependent, and the mechanism looks like a race:

- keys are **layer-1 states, which block 0's own training reshapes**. As block 0 learns context mixing,
  the same key token in the fact segment (after `v_{j-1}`) and in the query (after the query marker)
  drift apart, eroding the alignment identity init gave;
- memory only becomes useful — and the gate only opens — once alignment is good enough.

Seed 0 opened the gate before the drift won; seed 1 did not.

- **H5d — layer-0 keys survive the failing seed.** Token + position embeddings have no block-0 context to
  drift. `k0` on seed 1 lifts off.
- **H5e — a gradient floor survives it.** With the gate shut, the read path gets no gradient to track the
  drift. `--read-grad-floor 0.1` on seed 1 lifts off.

## Runs (8 facts, 4000 steps; seed 0 unless noted — comparable to r3_f8's 83.9%)

| order | name | change from r3_f8 |
|---|---|---|
| 1 | r5_k0 | `--key-layer 0 --read-layers 0` |
| 2 | r5_k0_s1 | same, **seed 1** (H5d) |
| 3 | r5_floor_s1 | `--read-grad-floor 0.1`, **seed 1** (H5e) |
| 4 | r5_bg16 | `--beta-grad 16` |
| 5 | r5_bg16_k0 | both |

(Original table, superseded:)

| name | change from r3_f8 |
|---|---|
| r5_bg16 | `--beta-grad 16` |
| r5_k0 | `--key-layer 0 --read-layers 0` |
| r5_bg16_k0 | both |
