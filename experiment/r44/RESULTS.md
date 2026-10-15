# r44 — 8 facts asymptotes near 80%, ~10 points short of the four-fact ceiling

Does more training close the last gap to 90.3%? 24k steps, warm from the 8-fact curriculum checkpoint.

| steps | exact | char | pos0 | pos1 | gate |
|---|---|---|---|---|---|
| 6k (`r38_curr8`) | 46.4% | 67.7% | 55.9% | 79.4% | 0.487 |
| 12k (`r43_curr8_12k`) | 73.6% | 84.9% | 83.6% | 86.2% | 0.828 |
| **24k (`r44_curr8_24k`)** | **78.6%** | 88.1% | 86.6% | 89.7% | **1.019** |
| 4-fact ceiling (`r37_f4`) | 90.3% | 95.1% | 94.1% | 96.1% | 0.275 |

## Verdict

**Diminishing returns, asymptote near 80%.** 46 -> 73.6 (+27 over 6k) -> 78.6 (+5 over the next 12k). The gate is now
saturated (>1.0), pos0/pos1 both near 88%. More steps will not reach 90.3%: there is a **residual ~12-point gap
between 4 and 8 facts that training does not close.**

## Diagnosis

The gap is the char-level multi-token-key tax (r36) scaling with fact count. At 4 facts the model disambiguates the
shared `:` key nearly perfectly (94% pos0); at 8 facts, even fully trained with the gate wide open, pos0 tops out at
~87%. Every extra fact adds another near-identical delimiter slot for the query to confuse, and the curriculum can
push that back but not to zero. This is the clearest evidence yet that the residual language limit is the addressing
constraint, not training or capacity — and it is exactly what a single-token (word-level) key would remove.

## What this does NOT show

- The asymptote's exact value (78.6% at 24k, still creeping — best 78.6). Called "~80%".
- One seed (but r45 shows the 12k result is seed-robust: 73.6 / 73.8).
