# r63 — RESULTS: the multi-layer escape PARTIALLY transfers to real language

r59 showed multi-layer read+write escapes the fact-count cliff cold on synthetic MQAR (9% → 99%). Does it
transfer to real language (BPE)? Cold, single-layer vs read+write@all, 16/32 facts, r50 BPE config.

## Recall (BPE, cold, final)

| n_facts | single-layer (read@0/write@last) | multi-layer (read+write@all) | lift |
|---|---|---|---|
| 16 | 53.4% | **86.9%** | +33.5 |
| 32 | 16.1% | **50.0%** | +33.9 |

## What it shows
- **The escape transfers as a large partial lift.** Multi-layer read/write adds ~+33 points at both fact
  counts — the single layer-0 read/write site is a real bottleneck on real language too, and opening
  every layer relieves it, just as on MQAR.
- **But it does NOT fully escape** (86.9% at 16f, 50.0% at 32f — vs MQAR's 99/96). So BPE has a **second
  bottleneck** the placement fix cannot clear: the multi-token / embedding-space-competition tax (r36 the
  delimiter/multi-token key; r47 real word-keys competing with the prose embedding space). Single-layer
  BPE cold is also less catastrophic than MQAR (53/16 vs 9/4), consistent with BPE's cliff being a
  different, milder mechanism than MQAR's pure bootstrapping deadlock.

## Verdict
- **H63 (multi-layer escapes the BPE cliff cold):** PARTIALLY CONFIRMED. Large curriculum-free lift
  (+33 pts) but not the full escape MQAR showed. On real language you need BOTH multi-layer placement
  AND the curriculum (r49) for 100% — they attack different bottlenecks (single-site vs embedding
  competition).

## Opens
- Multi-layer placement + curriculum combined (does the lift compound to 100% faster than curriculum
  alone)? Not run — the two escapes are now separately characterised.
- This is the read/write PLACEMENT; the CSA2 shared-store build (r64+) is a separate axis (cheap
  retrieval), not another escape.
