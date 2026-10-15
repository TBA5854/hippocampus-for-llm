# r46 — the char-level language curve, finished

The last two points, both warm-started with the curriculum.

| run | facts | steps | exact | char | pos0 | gate |
|---|---|---|---|---|---|---|
| `r46_curr12_24k` | 12 | 24k | **62.3%** | 78.5% | — | 1.34 |
| `r46_curr13_12k` | 13 (task cap) | 12k | 54.8% | 73.8% | 59.4% | — |

## Verdict

- **12 facts keeps climbing with steps** (r45's 54.5% @12k -> 62.3% @24k), same diminishing-returns shape as 8 facts
  (r44). **13 facts** — the task's hard cap (2 distinct letters/key from 26) — reaches 54.8% at only 12k steps.

## The complete char-level language curve (facts in text, out of context, from the store)

| n_facts | best exact | notes |
|---|---|---|
| 4 | 90.3% | near-ceiling, controlled (shuffle 0.2%) |
| 8 | 78.6% | asymptote ~80% (r44), seed-robust |
| 12 | 62.3% | still climbing at 24k |
| 13 | 54.8% | task cap, only 12k steps |

Monotone decline with fact count, every point far above chance and its scrambled control. The curriculum is the
lever throughout; the decline is the char-level multi-token-key tax (r36/r44) compounding per fact — the tax r47
shows BPE removes.

## What this does NOT show

- 13 facts at 24k (only 12k run) — likely climbs like 12 did.
- Beyond 13 facts is impossible in the char task by construction; BPE (r47+) is how the fact count goes higher.
