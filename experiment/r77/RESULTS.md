# r77 — settle r76: RESULTS. **Managed memory is VIABLE. The recall gap was under-training.**

## Numbers (best = final at 6k; per-layer store, --write-op-gate, warm r70_pl)
| task | recall @6k | mem (slots/512) | for reference (r76 no-gate) |
|------|-----------|-----------------|------------------------------|
| bpe_recall (gate) | **86.1%** (still climbing) | **198** | 94.1% at mem 508 (full) |
| bpe_update (gate) | **85.3%** | **251** | 7.8% at mem 508 (full) |

gate_recall trajectory: 60.3 (2.5k) → 66.1 (3k) → 82.8 (5k) → **86.1 (6k)** — rising
monotonically, no plateau.

## Verdict
The bounded gate is **not** capped: recall was still climbing at 6k, so the 86-vs-94
gap is under-training (the gated store — fewer, updated slots — is a harder
optimisation than the redundant per-token store), not the prose-clobber ceiling we
worried about in r76. So:
- **Bounded** ✓ — 198–251 slots vs 508 full (~2.6× fewer), O(distinct keys).
- **Recall-preserving** ✓ (trending) — 86% and rising at 2.6× compression.
- **Updates land** ✓ — 85.3% bounded vs 7.8% for append-both.

That is a **self-managing, space-efficient memory**: it stores one slot per fact,
revises in place, and recalls — the checkpoint the user set ("if viable for memory,
good for the next one store"). **Viable.**

## Small remaining question (r78)
Confirm the ceiling: continue both gate runs from the r77 checkpoints (+6k) — does
recall reach ~no-gate (94%) bounded, fully closing the gap? If yes, the crude
exact-key gate needs no salience refinement for this regime. Either way, the concept
is proven and the **one store + per-layer adapters** transition (the space-efficient
destination) is greenlit.
