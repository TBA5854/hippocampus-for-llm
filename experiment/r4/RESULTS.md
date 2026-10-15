# r4 — results: persistence works; the 8-fact result does not replicate across seeds

## Training runs (8 facts, 4000 steps, r3's same-space placement)

| run | seed | best | final | verdict |
|---|---|---|---|---|
| r3_f8 (from r3) | 0 | 83.9% | **83.9%** | lifted off at step ~1750 |
| r4_f8_s1 | 1 | 6.1% | **3.9%** | **chance** |
| r4_f8_s2 | 2 | 4.1% | **3.6%** | **chance** |
| r4_f8_off (HC off) | 0 | 4.1% | **2.5%** | chance — control holds |
| r4_f16_12k | 0 | — | — | stopped by design at start (see below) |

Chance 3.1%.

```
alignment on identity slots (offset 0), seed 1:  30 -> 11 -> 7 -> 0% by step 1000, gate 0.005 throughout
alignment on identity slots (offset 0), seed 2:  13 -> 13 -> 4 -> 0% by step 1000, gate 0.010 throughout
```

## Verdicts

- **H4a (not a seed accident) — falsified.** 1 seed of 3 works. The honest 8-fact headline for r3's
  config is **"83.9% on one seed, chance on two"**, not 83.9%.
- **H4b (HC off at chance) — confirmed.** 2.5%. With r3's scrambled 2.7%, both controls hold: when this
  config works, it works through stored content.
- **H4c (16 facts is budget) — not tested.** Stopped 30 s after start: on a placement that fails 2 seeds
  in 3 at 8 facts, a 16-fact budget run is confounded by the same failure. Rerun on r5's placement.

## Diagnosis — why seeds diverge

Both failing seeds show the same trajectory: identity-init alignment is **present at the start and decays
to zero before the gate ever opens**. Seed 0 opened its gate first and survived.

The keys are layer-1 states, so they are a function of block 0, and block 0 is being trained by the
language-model loss the whole time. As it learns context mixing, the same key token in two contexts —
after `v_{j-1}` in the fact segment, after the query marker in the query segment — maps to increasingly
different states, and the alignment identity init provided erodes. Memory only becomes useful (and the
gate only opens) once alignment is good enough. It is a race, and 2 of 3 seeds lose it.

This is an inference from alignment and gate trajectories, not a measurement of cross-context
similarity; that mechanistic check is deliberately not being chased here (standing note: mech-interp is
deferred). r5 tests the explanation behaviourally instead, on the failing seed:

- keys at **layer 0** (token + position embedding) have no block-0 context to drift;
- a **read-path gradient floor** lets the query track drift while the gate is still shut.

## Offline (on r3_f8's checkpoint — the seed that works)

**Persistence: done.** `mech/persist.py`, as two separate processes: process 1 writes the fact and
distractor segments into the store, saves it with the model's weight fingerprint, and exits; process 2
loads the checkpoint and the store and answers a query segment that contains no facts.

| | recall | n |
|---|---|---|
| in-process | 81.25% | 160 |
| **reloaded in a fresh process** | **81.25%** | 160 |

Identical, so nothing the store needs is being lost between processes (including the centring mean,
which writes update at eval time and so has to travel with the file). Store: ~1.05 MB per example at
M=512, d=256 (384 slots used). A foreign model's fingerprint is refused (tested).

**Miss taxonomy: every matching miss is another fact's key.**

| read lands on | 8 facts | 2 facts |
|---|---|---|
| correct slot | 81.9% | 88.8% |
| **another fact's key** | **18.1%** | **11.3%** |
| a value slot | 0% | 0% |
| distractor text | 0% | 0% |
| query marker | 0% | 0% |

(Right-slot-wrong-answer is 0.3% / 0.9%: content is solved.) Distractor slots — two thirds of the store —
never attract a read, which is encouraging for larger stores. The failure is confined to telling one key
symbol from another.

## Objective status after r4

| requirement | status |
|---|---|
| read early, write @ last block | done |
| >= 8 facts out of context | **1 of 3 seeds** (83.9%); not yet a robust result |
| controls at chance | **done** — HC off 2.5%, scrambled 2.7% |
| store persisted + reloaded | **done** — 81.25% == 81.25% across processes |
