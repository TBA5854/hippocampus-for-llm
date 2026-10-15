# r87 — gate-free width×depth: the r85 cold-bootstrap failure is the GATE, not depth

**Gate-free (plain store, no write-op gate), ALL four shapes cold-bootstrap the store,
including the deep ones that flatlined WITH the gate. So r85's "depth kills cold-bootstrap"
is really a GATE×DEPTH interaction — the plain store bootstraps at any depth; the write-op
gate is what cannot be learned from scratch in a deep stack.**

## Numbers (bpe_recall, cold, 32 facts, M=1024, 12k, plain store vs r85 gated)
| shape (d / layers) | gate-free (r87) | gated (r85) |
|--------------------|-----------------|-------------|
| d512 / 2  | **98.8%** (best 98.9) | 96.7% |
| d384 / 3  | **97.5%** | 95.3% |
| d192 / 12 | **97.0%** | **0.2%** |
| d256 / 6  | **84.7%** | **0.3%** |

## Reading
- The two shapes that FAILED with the gate (d256/6L at 0.3%, d192/12L at 0.2%) both
  **cold-bootstrap fine without it** (84.7%, 97.0%). Depth is not the barrier: a 12-layer
  model reaches 97% on the plain store. What r85 measured is the **gate × depth**
  interaction — the write-op controller (new/update/skip) adds a learning burden the read
  gate cannot get past cold in a deeper stack, so the store is written but never read.
- The shallow shapes are ~unchanged (the gate was learnable there): d512/2L 96.7→98.8,
  d384/3L 95.3→97.5.
- Odd note: the "square" d256/6L is the weakest gate-free too (84.7 vs ~97 for the others,
  still climbing at 12k) — 6 layers at this width bootstraps the plain store slowest, for
  reasons not chased here (no mech-interp dive).

## Verdict
Corrects the r85 headline: **the managed (gated) store, not depth, is what fails to
cold-bootstrap in a deep stack.** The plain store is depth-robust. This reframes the open
question as "how to bootstrap the GATE" — a curriculum (warm the gate from a converged
plain-store model) is the obvious fix and is r90. Folded into the paper's scale-up
narrative and limitations.
