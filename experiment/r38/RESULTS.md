# r38 — the curriculum escapes the language fact-cliff; the 4-fact result is controlled

| run | exact | char | pos0 | pos1 | gate |
|---|---|---|---|---|---|
| `r37_f4` — 4 facts cold (baseline) | 90.3% | 95.1% | 94.1% | 96.1% | 0.275 |
| `r38_f4_shuf` — 4 facts SHUFFLED (control) | **0.2%** | 4.0% | 4.5% | 3.4% | 0.047 |
| `r36_lang_s1` — 8 facts cold (baseline) | 10.3% | 45.9% | 11.4% | 80.5% | 0.206 |
| **`r38_curr8` — 8 facts WARM from 4-fact ckpt** | **46.4%** | 67.7% | **55.9%** | 79.4% | **0.487** |

## Verdicts

- **H38a — the 4-fact result is real. Confirmed.** The matched `--shuffle-store` control sits at chance on all three
  metrics (exact 0.2%, char 4.0%, pos0 4.5% vs chance 3.8%), so 90.3% is retrieval, not a task leak. This is the
  control r37 lacked; the 90.3% headline now stands.
- **H38b — the curriculum breaks the 1/n_facts ceiling. Confirmed.** Warm-starting 8 facts from the 4-fact
  checkpoint takes `pos0` from 11.4% (a uniform draw over 8 facts, = 1/n) to **55.9%**, exact from 10.3% to
  **46.4%**, and opens the gate from 0.206 to 0.487. The model learns to disambiguate the shared `:` key once it is
  shown the solvable 4-fact case first.

## Diagnosis — this is r25's escape, transferred to language

r24/r25 established on MQAR that the fact-count cliff is **not a capacity limit** but a gate/decoder bootstrapping
deadlock, escapable by warm-starting from a solved easier setting (16 facts 3.8%->100%, 48 facts 0.9%->100%). r37
found the same cliff in the text vocabulary (4->8 facts). This round shows the same escape works: the language cliff
is bootstrapping, not a wall. It does not reach the 90% ceiling in one 4->8 step (r43 tests whether more steps /
a longer ladder close the gap), but the mechanism is confirmed.

## What this does NOT show

- One seed for the warm start (r40 adds seed 2, and it landed higher — 69.4%).
- H38c (shift-3 x curriculum compose) needed `r38_f4_s3`, which finished after this write-up; measured in r40.
- 8 facts is below the task's 13-fact cap; the ladder above 8 is r40.
