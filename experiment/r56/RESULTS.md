# r56 — RESULTS: on a shared store, placement doesn't matter — the 2×2 matrix is a flat tie

The user's msg 30/39 ask, done on one controlled setup: read @ start vs @ every layer × write @ last vs
@ every layer. r41 had settled only the read half (on the char task); this is the full matrix on MQAR
(8 facts, r11 config, cold), with the write-side plumbing (`--write-layers`) built this round.

## The matrix (recall; shared single store, key @ layer 0)

| | write @ last | write @ every layer |
|---|---|---|
| **read @ layer 0** | **99.8%** (`r56_wlast`) | **100.0%** (`r56_wall`) |
| **read @ every layer** | **99.2%** (`r56_rall_wlast`) | **100.0%** (`r56_rall_wall`) |

Replicates (seed 1): write@all 100.0%, read@all+write@all 100.0%. Scrambled controls: 3.3% (chance).

## What it shows
Every cell is 99.2–100% — read placement and write placement are **statistically tied**, and the
scrambled controls at chance confirm the recall is real, not a parameters-alone artifact. So:
- The objective's minimal **read-early / write-late** placement leaves nothing on the table.
- Reading and/or writing at **every** layer buys nothing (a hair *below* on read@all+write@last, 99.2
  vs 99.8 — the r41 direction, but here inside noise).
- **Why:** the key is the layer-0 token embedding in every cell (`--key-layer 0 --key-pos-free`), so the
  matching that drives recall is identical regardless of where the value is read or written. The value
  content is complete enough at any of these depths (the block-0 value-skip already carries early
  content into the last-block write), so averaging more depths in (`write@all`) neither helps nor hurts.

This is slightly stronger than r41's read-only result (88.3 vs 90.3 on the char task): on the cleaner
MQAR task everything saturates at ~100%, so placement differences wash out entirely.

## Verdict
- **H56 (write@all does not beat write@last; symmetric r/w@all does not beat r-early/w-late):** CONFIRMED.
  Flat tie across the 2×2, controls at chance. The single last-block write + early read is the right
  minimal choice — matching is a property of the layer-0 key, not of where the value lives.

## The caveat this raises (→ r57)
Every cell here uses ONE shared store addressed by a single layer-0 key. The flat tie may just say "the
layer-0 key dominates, so nothing else you do matters." The real test of "memory at every layer" is a
**unique store per layer** — each depth keyed and valued at its own level, no layer-0 bottleneck. r57
builds that and pushes it, and the shared store, to the fact-count breaking point.
