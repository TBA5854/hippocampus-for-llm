# r25 — is the fact count a capacity limit, or another lift-off cost?

## Ideate (from r21, r23, r24)

The evidence is pulling in one direction:

| observation | round |
|---|---|
| 32 facts at the 8x store: **100%** — but warm-started from an 8-fact model | r21 |
| 128 symbols, 8 facts, **cold**: 99.7%, lift-off at ~4500 steps (3x slower than 32 symbols) | r23 |
| 128 symbols, 48 facts, **cold**: 0.9% at 8000 steps, read mass decaying 0.97 -> 0.70 | r23 |
| 128 symbols, 16 facts, **cold**: ~5% at 6500 steps, gate open but flat, mass healthy at 0.99 | r24 |

Every store-size failure had the same shape and the same fix: cold runs never start using the read, and a warm start
from the size below solves it (r19, r20, r22 — 4x, 8x, 16x all reached 100% that way). The fact count now shows the
same signature. 16 facts is not decaying like 48 — its mass stays at 0.99 and its gate keeps opening — it simply has
not lifted off within the budget.

So the natural question is whether "more facts" is a *capacity* limit at all, or the same lift-off cost in another
guise. The store itself is clearly not full: 508 slots hold 48 facts trivially, and 4,064 slots held 32 facts at 100%.

## Hypotheses

- **H25a — a fact-count curriculum works.** Warm-starting 16 facts from the trained 8-fact model
  (`r23_1x_sym128_f8.pt`, same 128-symbol vocabulary, so the shapes match) reaches >= 95% within 4000 steps, where the
  cold run sat at ~5% after 6500. Falsified if it stays near chance: then the fact count is a real capacity limit and
  the curriculum only ever bought store size.
- **H25b — the curriculum continues to 48 facts**, warm-started in turn from the 16-fact model, >= 95%.
- **H25c — the matched scrambled control is at chance** at 48 facts (0.8% chance at 128 symbols).

## Falsifiers, watched at every eval

- Read mass decaying below ~0.9 -> dilution, i.e. genuine interference between facts rather than a slow start.
- slot@1 dropping below ~100% -> a matching failure, the first in many rounds; that would be a different story from
  everything since r9.

## Runs (1x store, 128 symbols, 8000-step budget for the cold reference, 4000 for warm starts)

| box | name | change |
|---|---|---|
| box2 | r25_1x_f16_warm | `--n-facts 16 --init experiment/r23/ckpt/r23_1x_sym128_f8.pt`, 4000 steps |
| box2 | r25_1x_f48_warm | `--n-facts 48 --init experiment/r25/ckpt/r25_1x_f16_warm.pt`, 4000 steps |
| box2 | r25_1x_f48_warm_shuf | as above + `--shuffle-store` (matched control) |

Box1 continues the store-size line (r23's 16x control, then 32x if that lands).
