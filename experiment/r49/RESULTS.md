# r49 — the BPE curriculum ladder holds at 100% through 32 facts

Warm-started from the 100% 8-fact model (r48_bpe8_curr), 12k steps.

| facts | exact | answer CE | note |
|---|---|---|---|
| 8 (r48) | 100.0% | 0.034 | base |
| **16 (`r49_bpe16_curr`)** | **100.0%** | 0.001 | 99.7% by step 500 |
| **32 (`r49_bpe32_curr`)** | **100.0%** | 0.005 | 98.9% by step 500, one 8->32 jump |

## Verdicts

- **H49a — 16 facts warm reaches the ceiling. Confirmed.** 100.0% (cold was 52.8%, r48).
- **H49b — 32 facts warm lifts to the ceiling. Confirmed, and in ONE jump.** 100.0% from an 8-fact base directly
  (cold was 5.9%) — no intermediate 16-fact rung was needed. Both were already ~99% by step 500, so the warm start
  does almost all the work and the remaining steps just finish it.

## Diagnosis — scaling is solved, and it is curriculum-bound

The fact-count axis, the only scaling limit left after r39 (store size irrelevant) and r31d (width fixed), is now
beaten on real-language facts: **8 / 16 / 32 facts all at 100%**, stated in prose, queried out of context, answered
from the store. The two historical causes are both gone:

- the multi-token-key addressing tax (char-level) — removed by BPE single-token word-keys (r47)
- the fact-count bootstrapping deadlock (r25) — escaped by the curriculum, which ladders freely and cheaply (one
  warm-start per rung, and 32-from-8 skipped a rung)

So adding facts costs a warm-start, not a redesign, and the count is no longer vocabulary-bound (the char task
capped at 13). "Scaling" is now a training-schedule question, not an architecture one.

## What this does NOT show

- One seed per rung. The result is clean (CE ~0.001, unambiguous), but seed spread is unmeasured.
- The upper limit. 64+ facts, and whether the store's 512 slots eventually bind (at 32 facts x tiled pairs the store
  is not full), are untested.
- Persistence across a process restart on the BPE task — the objective's save/reload demo, done on MQAR but not yet
  on real-language facts. This is the last unbuilt objective piece and the obvious next round.
