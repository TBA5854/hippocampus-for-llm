# r57 — unique-per-layer store vs shared store, pushed to the fact-count breaking point

r56 answered the placement question for a SHARED store (one memory, addressed by a single layer-0 key):
write@last ≈ write@all. The user's follow-up: make the store **unique per layer** (each depth its own
memory, keyed and valued at that depth — removing the layer-0 key bottleneck), add it to the ablation,
and **push every variant to its breaking point**.

`--per-layer-store` (r57) is now built: each layer in `--write-layers` gets its own `EpisodicStore`,
keyed on the pre-read state at that layer and valued on its post-block state, read back at that layer.

## The breaking point

r24 found the SHARED cold cliff between 8 and 16 facts (8f 99.7% → 16f 3.8% → 32f 3.1%), a bootstrapping
deadlock (r25), with slot@1 = 100% throughout (never a matching failure — the read finds the right slot,
the content just isn't there yet cold). This round sweeps n_facts cold across the placement variants to
see where each breaks and whether the unique-per-layer store moves the cliff.

## Setup — MQAR cold, r11/r56 config, sweep n_facts

- **box1 — unique per-layer** (`--per-layer-store --write-layers 0..5 --read-layers 0..5`, no value-skip):
  `r57_pl_f8 / f16 / f32`, plus `r57_pl_f8_shuf` (scrambled control).
- **box2 — shared symmetric** (`--read-layers 0..5 --write-layers 0..5 --value-skip-layer 0`, one store):
  `r57_sh_f16 / f32 / f64` (f8 = 100% from r56_wall) — pushes the shared store past the cliff too.

## Hypothesis

**Claim:** the unique-per-layer store does NOT beat the shared layer-0-keyed store, and likely breaks
EARLIER. Its per-layer keys at depth i>0 are contextualised states (r42): the same fact token in
different prose gives a different key, so matching a fact seen across varying contexts is unreliable —
whereas the layer-0 key is the context-invariant token embedding. The frozen-identity query makes
layer 0 match exactly either way, so the shared store's single layer-0 key is doing the real work.

**Predict:** per-layer f8 ≤ shared f8 (99.7%) — possibly well below if deep stores inject noise; both
collapse by 16 facts cold (the r24 cliff is bootstrapping, not placement). Scrambled at chance.

**Falsified if** per-layer f8 clearly beats shared, or per-layer holds facts past 16 cold where shared
cliffs — either would say a per-depth memory carries bindings the single layer-0 key cannot, and the
"exact token key is load-bearing" thesis (r5/r42) is incomplete.

## What it decides
Completes the placement study with the unique-store arm the shared matrix (r56) could not cover, and
locates the cold breaking point for each. A per-layer loss confirms the layer-0 exact key is the
architecture's foundation; a per-layer win reopens where memory should live in the stack.
