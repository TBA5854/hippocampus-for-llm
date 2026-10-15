# r18 — results: the plain skip breaks between 1x and 2x; normalising fixes it at 2x

## Numbers (TinyStories, single-occurrence facts, 8 facts, one-shot read beta 32, write @ last block + block-0 skip,
## key @0 position-free, frozen identity query, batch 32, 8000 steps, seed 0)

| run | store | slots filled | skip | best | final | first >= 50% | first >= 95% | gate (final) | read mass min / final |
|---|---|---|---|---|---|---|---|---|---|
| r18_skip_M512 | 1x | 384 | plain | 100.0% | **100.0%** | 1750 | 3000 | 0.283 | 0.99 / 0.99 |
| r18_skip_M1024 | 2x | 896 | plain | 29.2% | **28.1%** | — | — | 0.088 | 0.82 / 1.00 |
| **r18_skipnorm_M1024** | 2x | 896 | **normalised** | 99.4% | **99.4%** | 5250 | 6750 | 0.179 | 0.82 / 0.98 |

Correct-slot reads 100% at every eval of all three.

## Verdicts

- **H18a (skip path works at 1x on TinyStories) — confirmed.** 100%. The corpus change did not break it; store size did.
- **H18b (plain skip works at 2x) — falsified.** 28.1% after 8000 steps (partial lift-off, still low).
- **H18c (normalising helps at 2x) — confirmed.** 99.4%.

## The objective placement across store sizes (TinyStories unless noted)

| store | plain skip | normalised skip |
|---|---|---|
| 1x | **100%** (r18; also 100% on tinyshakespeare, r11) | — |
| 2x | 28.1% | **99.4%** (lift-off ~5250) |
| 4x | chance at step 5000, stopped (r16) | chance through 12000 (r17) |

Block-0 store alone, for reference: 100% at 1x, 2x and 4x.

## Diagnosis

Two distinct effects:

1. **Norm balance matters, and matters more as the store grows.** At 1x the plain sum works; at 2x only the normalised sum
   does. So the last-block half does crowd out the block-0 half, increasingly with scale.
2. **Lift-off time for the last-block write grows steeply with store size.** Normalised skip: ~5250 steps to 50% at 2x vs
   ~1750 for the plain skip at 1x, and nothing by 12000 at 4x. The block-0 store's lift-off grows too (1250 -> 3750 from
   1x to 4x on TinyStories) but much less.

## Next

Rather than more steps from scratch at 4x, warm-start: initialise the 4x run from `r18_skipnorm_M1024`'s checkpoint,
which already uses the skip path at 2x. If the difficulty is discovering how to use the read (lift-off), a model that has
already discovered it should carry over, and the 4x store only has to be adapted to.
