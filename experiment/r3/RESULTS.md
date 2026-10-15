# r3 — results: 8 facts at 83.9%, written at the last block

## Numbers (4000 steps, sequential, seed 0)

Same-space placement: `--write-layer 5 --key-layer 1 --value-shift 0 --query-init identity`.

| run | facts | best | final | content-miss | align -1 / 0 / +1 |
|---|---|---|---|---|---|
| **r3_f8** | **8** | 83.9% | **83.9%** | 0.6% | 0 / **84** / 0 |
| r3_f8_shuf (scrambled) | 8 | 4.1% | 2.7% | — | — |
| r3_f2 | 2 | 89.7% | 88.3% | 0.6% | 0 / 88 / 0 |
| r3_f16 | 16 | 4.5% | 3.3% | 9.8% | 0 / 10 / 0 |
| r3_w5s1_align (r1 placement) | 2 | 80.2% | 26.7% | 0.0% | 3 / 0 / 6 |
| r3_w4_align (100% reference) | 2 | 100.0% | 99.8% | 0.0% | 0 / 0 / **100** |

Chance 3.1%.

```
f8     3  2  3  2  3  2  3 16 59 68 81 83 82 84 84 84
f2     2 44 58 69 75 80 81 81 83 83 85 90 88 89 88 88
f16    3  3  3  4  3  4  5  5  2  2  3  2  3  3  3  3
```

## Against the starting point

| | 8 facts, best | written at |
|---|---|---|
| old architecture (`dg8_off`: key from write block, shift 1, random query) | **4.1%** (chance) | block 4 |
| **same-space keys (`r3_f8`)** | **83.9%** | **block 5, last before the head** |

Scrambled control at matched settings: 2.7%. The gain is stored content, not added capacity. This is a
multi-variable change against the old run; the one-variable ablation that isolates the key idea is at 2
facts: layer-1 keys with random query init 58.1% (r2) -> identity init **88.3%** (r3).

## Verdicts

- **H3a (the cliff was matching) — confirmed at 8 facts.** 4.1% -> 83.9% in the same 4000-step budget.
- **H3b (clean at the objective's placement) — mostly.** The value comes from the last block, stably, no
  collapse. But 2 facts reaches 88.3%, not 100%.
- **H3c (prediction pairing) — confirmed**, and the instrument with it: the 100% reference reads at
  offset +1 in 100% of cases. `w5s1`'s alignment scatters (3/0/6) as it collapses: a matching failure.

## Diagnosis — what is left

1. **Content is solved.** Content-miss 0.6% at both 2 and 8 facts: when the read lands on the right slot,
   the last-block value is decoded correctly almost every time.
2. **The limit is matching, and training does not improve it.** Identity-slot reads ~84-88% end, ~90% at
   step 10 of the smoke test. Recall tracks alignment exactly, and alignment never climbs past where
   init put it. Where the other ~12% of reads go is unmeasured: wrong fact's key, a value slot, or
   distractor text. Those imply different fixes (key interference vs. distractor interference), so it is
   the next measurement, not a guess.
3. **16 facts is at chance, but that may be budget.** 8 facts also sat at chance until step ~1750 before
   lifting off in ~500 steps; 16 facts plausibly needs longer. Not distinguishable from a capacity limit
   with this data.

## Objective status

| requirement | status |
|---|---|
| read @ 1, write @ last block | **done** |
| >= 8 facts, out of context | **83.9%** single seed |
| controls at chance | scrambled 2.7%; HC-off at 8 facts on this config not yet run |
| store persisted + reloaded | **not started** |
| multi-seed | not started |
