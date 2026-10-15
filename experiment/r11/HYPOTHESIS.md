# r11 — make the last-block write robust on facts seen once

## Ideate (from r10)

r10's skip path — value = last-block state + block-0 output, write still at the last block — reached **93.9%** on
single-occurrence facts, and was still rising at step 4000 (74 -> 85 -> 91 -> 93 -> 94). Its checkpoint persists:
**94.06% reloaded in a fresh process == in-process, mismatched store 3.4%** (n=320). That is the objective's full
placement (read @ first layer, write @ last block) storing and persisting facts the model cannot predict.

What it has not had: a budget that lets it finish, a second seed, its own scrambled control, and a scale test. The
block-0 store it borrows from already passed all of those in r10 (seed 1 100%, scrambled 2.3%, 16 facts 100%,
32 facts ~100%).

## Hypotheses

- **H11a — budget.** At 8000 steps the skip path reaches >= 99% (r10 was still climbing at 4000).
- **H11b — replicates.** Seed 1 reaches >= 95%.
- **H11c — control.** Scrambled store at chance.
- **H11d — scales.** 32 single-occurrence facts stays >= 95%.

## Falsifiers

- Plateau below ~95% at 8000 -> the last-block half of the sum degrades the observed token; next: normalise the
  two halves before adding, or a learned mix.
- Seed 1 fails -> lift-off is fragile for the skip path (it lifts off late, ~step 2250, which is the regime where
  earlier configs were seed-sensitive).
- Scrambled above chance -> leak.

## Runs (single occurrence, 8000 steps; key @0 position-free, frozen identity query; write @5 + skip @0, shift 1)

| order | name | facts | seed |
|---|---|---|---|
| 1 | r11_skip_s0 | 8 | 0 |
| 2 | r11_skip_s1 | 8 | 1 |
| 3 | r11_skip_shuf | 8 (scrambled) | 0 |
| 4 | r11_skip_f32 | 32 | 0 |

~22 min each, ~90 min total.
