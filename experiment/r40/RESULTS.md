# r40 — the curriculum ladder continues, replicates, and composes with shift-3

| run | exact | char | pos0 | pos1 | gate |
|---|---|---|---|---|---|
| `r37_f12` — 12 cold (baseline) | 6.6% | 42.6% | 7.8% | 77.3% | 0.147 |
| **`r40_curr12` — 12 WARM from 8-fact curr** | **41.6%** | 61.6% | **52.3%** | 70.9% | 0.659 |
| `r37_shift3` — 8 shift-3 cold (baseline) | 31.6% | 55.1% | 62.0% | 48.1% | 0.130 |
| **`r40_s3curr8` — 8 shift-3 WARM (compose)** | **46.6%** | 66.6% | 60.8% | 72.5% | 0.475 |
| `r38_curr8` — 8 warm, seed 1 | 46.4% | 67.7% | 55.9% | 79.4% | 0.487 |
| **`r40_curr8_s2` — 8 warm, seed 2 (replication)** | **69.4%** | 81.9% | 81.1% | 82.7% | 0.696 |

## Verdicts

- **H40a — the ladder continues. Confirmed.** 12 facts warm from the 8-fact curriculum checkpoint reaches 41.6%
  (pos0 7.8%->52.3%), where 12-cold was 6.6%. The escape stacks step by step, exactly as r25's 16->48 ladder did on
  MQAR.
- **H40b — shift-3 and the curriculum compose, but do not stack much. Partially confirmed.** Warm shift-3 (46.6%)
  beats cold shift-3 (31.6%) and matches the plain curriculum (46.4%). So the two levers address *overlapping*
  bottlenecks and the curriculum is the dominant one: adding shift-3 on top of the curriculum buys little. What
  shift-3 does change is the *shape* — it lifts pos0 while the curriculum lifts pos1, and warm shift-3 gets both
  (pos0 60.8, pos1 72.5), the most balanced profile in the round.
- **H40c — the curriculum replicates. Confirmed, and then some.** Seed 2 reaches **69.4%** against seed 1's 46.4%.
  Both are far above cold's 10.3%, so the escape is robust; the magnitude carries the ~1.5x seed spread this
  architecture shows everywhere (r34: 3.3x, r32: on MQAR).

## Diagnosis

The curriculum is the load-bearing lever for language recall past the fact-cliff. Shift-3 (a task-layout change) and
the curriculum (a training-schedule change) both help, but they are not independent gains — both are ways of getting
the gate to open on a disambiguated read, and once the curriculum has done that, the layout matters less. The ladder
result (12 facts to 41.6%) plus the seed-2 result (8 facts to 69.4%) together say the language task is climbing the
same curve MQAR did, just from a lower, noisier start.

## What this does NOT show

- The ceiling. 69.4% (best seed, 8 facts) vs 90.3% (4 facts) — r43 tests whether 2x steps closes it.
- Ladders past 12 facts are blocked by the task's 13-fact cap (2 distinct letters/key from 26). A larger alphabet
  or word-level keys would be needed to go further.
