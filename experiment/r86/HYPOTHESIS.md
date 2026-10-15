# r86 — Memory transplant: stateless weights, stateful store

## Claim
The weights are stateless; the store is the state. Two consequences, tested directly:

1. **Same weights, swapped store = swapped memory.** One frozen model with two stores
   (two disjoint fact sets) answers each store's queries at full recall and the *other*
   store's queries at chance. Swapping the store swaps the memory with no weight changed.
   *(Already shown on the r85 `store_f64` gated model: own 100%/93.8%, swap 0.0%,
   scrambled 0.0%. r86 re-confirms on a clean matched pair and adds the foreign half.)*

2. **Different weights = the store does not read back.** A store's keys are the writing
   model's token embeddings; retrieval is an exact match against the reading model's
   query. So a store written by model B, force-loaded into model A (same shape, facts
   physically present, fingerprint check bypassed), should collapse to **chance** —
   memory is portable across a store-swap on the SAME weights but bound to the weights
   that wrote it.

## Prediction
- Same model: diagonal recall high (~99%), off-diagonal + scrambled at chance.
- Foreign: A←B and B←A recall at chance, despite each model reading its OWN store at ~99%.
- If the foreign transplant instead reads back well, that falsifies the model-specificity
  claim (memory would be a portable artifact) — written up either way.

## Method
Two INDEPENDENT models (different seeds, so different token embeddings), matched shape so
their stores are shape-compatible: `d256 L6`, `bpe_recall`, 16 facts, `M=512`, plain store
(value-skip-norm, NO write-op-gate — so both cold-bootstrap cleanly per r66), 12k steps.
Then `mech.persist swap` builds two stores per model and evaluates the full matrix:
same-model own/swap/scrambled, and foreign A↔B (in-process transplant, fingerprint bypassed).

## Runs (box1 after r85 drains)
- `r86_modelA` : bpe_recall d256L6 16f M512 seed 0, 12k, --save
- `r86_modelB` : bpe_recall d256L6 16f M512 seed 1, 12k, --save (independent)
- swap eval    : `persist swap --run/-ckpt A --run-b/--ckpt-b B --out results/swap.json`

Controls: scrambled-store (chance) built into the swap eval; both models must clear ~99%
on their OWN store first (else the foreign 0% is uninformative).
