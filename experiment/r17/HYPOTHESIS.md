# r17 — a normalised skip for the last-block write at the 4x store

## Ideate (from r16)

The last-block write with a plain block-0 skip (value = last + block0) worked at 512 slots (r11: 100%) but at the 4x
store on TinyStories stayed at chance for 5000 steps: gate closing (0.020 -> 0.010), read mass falling (0.54),
matching 100%. The block-0 store alone at the same size reached 100% (r15). So the block-0 half carries what is needed
and something about adding the last-block half prevents the read path from using it at scale. The simplest suspect is
norm: an unnormalised sum lets whichever half is larger dominate.

`--value-skip-norm` stores `mean|skip| * (unit(last) + unit(skip))`: equal weight to both halves, rescaled to the skip's
typical norm (commit dbf9133; tested against the formula).

## Hypotheses

- **H17a — normalising the halves lets the last-block write lift off at 4x.** >= 95% within 12000 steps, with the gate
  rising rather than closing.
- **H17b — the 4x store's scrambled control is at chance** (carried over from r16).

## Falsifiers

- Gate still closes and recall stays at chance -> not a scale problem; the last-block half is actively harmful at scale.
  Next: weight the skip higher than the last half, or accept that at scale the value should come from block 0 and the
  last block should contribute only through a learned gate.
- Lifts off but plateaus well below the block-0 store -> equal weighting is not enough; learned mix.

## Runs (single occurrence, 8 facts, TinyStories, one-shot read beta 32, M=2048, 16 seg, 16 x 2, seed 0)

| order | name | value | steps |
|---|---|---|---|
| 1 | r17_skipnorm_M2048 | norm(last) + norm(block 0), write @ last block | 12000 |
| 2 | r17_obs_M2048_shuf | block 0, scrambled store | 6000 |

~2.3 h + ~1.1 h.
