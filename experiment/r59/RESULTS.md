# r59 — RESULTS: the fact-count cliff is the SINGLE layer-0 bottleneck; multi-layer read/write escapes it cold

r57/r58 ran in the 32-symbol easy regime (no cliff). This is the real sweep at **128 symbols** (r24's cliff
regime), three placements, MQAR cold, single seed each.

## The breaking-point curve (recall = final)

| n_facts | single-layer (read@0, write@last) | shared-symmetric (read+write @all, ONE store) | per-layer store (6 stores) |
|---|---|---|---|
| 8  | 99.7% (r23) | — | 100.0% |
| 16 | **9.1%** ⬅ cliff | **99.4%** | **92.7%** |
| 24 | — | — | 71.6% |
| 32 | **4.2%** ⬅ cliff | **96.2%** | 100.0% |

Chance ≈ 1/128. Single-layer (the objective placement) reproduces r24's collapse: 99.7% at 8 facts →
9.1% at 16 → 4.2% at 32.

## The finding
**The r24 fact-count cliff is a property of the single layer-0 read/write site, not of the store or the
task.** Give the model read AND write access at every layer and it escapes the cliff **cold, with no
curriculum**:
- **shared-symmetric** (read@all + write@all into one store): 99.4% / 96.2% at 16 / 32 facts — clean.
- **per-layer** (a store per depth): 92.7% / 71.6% / 100.0% at 16 / 24 / 32 — escapes, but noisy.

vs single-layer's 9.1% / 4.2%. So multi-layer read/write is a **third escape from the bootstrapping
deadlock** (r25 curriculum was the first; store-size warm-start the second). More read/write sites =
more gradient paths for the gate/decoder to bootstrap disambiguation, so it clears the cold deadlock
that one layer-0 site cannot.

## This qualifies r56
r56 found placement was a **flat tie** — but that was 8 facts, the easy regime where everything
saturates at 100%. At the **breaking point** placement is decisive: multi-layer read/write is the
difference between 9% and 99%. "Placement doesn't matter" holds only below the cliff; at and above it,
reading/writing at every layer is what carries recall.

## Per-layer vs shared-symmetric — the user's bet
The bet ("per-layer holds longer") is **half-confirmed**: per-layer *does* escape the cliff where
single-layer dies (92.7% vs 9.1% at 16 facts). But it does **not** clearly beat shared-symmetric — and
it is markedly **noisier** (non-monotonic 92.7 → 71.6 → 100.0 across 16/24/32, a single-seed artifact
near the cliff; r45 saw the same seed variance in this regime). Shared-symmetric holds more cleanly
(99.4 / 96.2) at a **fraction of the cost** (one store, not six; per-layer runs were ~2× slower).

## Verdict
- **The cliff is the single-site bottleneck; multi-layer read/write escapes it cold:** CONFIRMED.
- **Per-layer beats shared:** NOT SUPPORTED — both escape; shared-symmetric is cleaner and cheaper. So
  a *single shared store read/written at every layer* is the better, simpler design — which **leans
  pro-CSA2** (share one KV state across layers rather than keep private per-layer memories).

## Caveats / opens
- Single seed per cell; the per-layer non-monotonicity needs 2–3 seeds to pin down (not re-run yet —
  the qualitative escape is unambiguous). Shared-symmetric would be the arm to replicate first.
- Not tested on real language (BPE); this is MQAR. The escape should be re-checked there.
- Does NOT bear on the north-star's sparse-read half — the read here is still a dense scan over all M.
