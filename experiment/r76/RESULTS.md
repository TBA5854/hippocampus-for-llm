# r76 — gated memory ops on the per-layer store: RESULTS

**Verdict: the managed-memory CONCEPT is viable — the store bounds itself and
updates land in place — but the crude exact-key rule's recall is not yet settled
(the recall arm was still climbing at the step trim). Partial green.**

## Numbers (final; recall = latest-value recall on the update task)
| task | arm | recall | mem (slots/512) |
|------|-----|--------|-----------------|
| bpe_recall | **gate** | 66.1% (still climbing) | **198** |
| bpe_recall | no-gate | 94.1% | 508 (full) |
| bpe_update | **gate** | **51.2%** | **251** |
| bpe_update | no-gate | 7.8% | 508 (full) |

## What holds
- **Bounded memory — confirmed.** The gate keeps occupancy at ~200–250 slots vs the
  no-gate's 508 (ring full). It genuinely stores ~one slot per distinct key =
  O(facts), not O(tokens). The space thesis, realised at the write side.
- **Updates need the gate — confirmed.** On the update task the gate (overwrite in
  place) hits 51.2% vs the no-gate's 7.8% (append-both blends old+new to noise). The
  update op works and is essential.

## What's unresolved
- **Recall under the gate is inconclusive.** gate_recall was **still rising** when
  the run stopped (16.9 → 39.2 → 60.3 → 66.1% across 3k warm steps; the warm-start
  dropped it because the gated store is a distribution shift, then it recovered).
  The 66-vs-94 gap is therefore *confounded with under-training* — we trimmed to 3k
  for the write-cost budget. Whether the crude gate reaches ~94% or plateaus below is
  open.
- **A known risk if it plateaus: prose-clobber.** The rule updates on *any* key-token
  collision. A fact's key token also appears in the prose distractor, followed by an
  arbitrary word — so a prose occurrence can overwrite the fact's slot (value_shift
  stores the *next* token). Blind dedup-by-token conflates a token's fact-role and
  prose-role. If recall plateaus below no-gate, this is why, and it motivates a
  *salience-aware* gate (don't overwrite a fact with filler).

## Perf note
First implementation rebuilt the whole [B,M,D] store out-of-place per position
(S×layers) — ~8× slowdown. Fixed by cloning the store once and doing in-place index
writes on the clones (safe vs the segment's read graph), → 0.64 s/step.

## Next (r77)
Settle viability cleanly: **run the crude gate to convergence** (8k) on recall and
update. If gate-recall reaches ~no-gate while staying bounded → **fully viable →
greenlight one store + adapters**. If it plateaus below (prose-clobber cap) → the
controller needs a **learned / salience gate** (r78) before the one-store step.
Either way the concept (bounded + updates-land) already holds.
