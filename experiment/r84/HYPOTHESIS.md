# r84 — route A + `write_skip_pad`: does removing the intermediate collision land CoT-over-memory?

## Where r83 left it
r83 established that route A (the CoT token scratchpad) does **real** retrieval on
every hop — the r81 "failure" was a masking bug. But 2-hop joint capped at ~38% with
**hop1 ≈ 90%, hop2 ≈ 45%**, and `hop2 ≈ 0.5 × hop1` is the tell:

The intermediate token `t1` is the **value** of `t0→t1` and the **key** of `t1→t2`.
With value-shift, t1's value-occurrence (in `t0 t1 pad`) writes a spurious slot
`t1→pad`; its key-occurrence (in `t1 t2 pad`) writes the real slot `t1→t2`. The
write-op gate keeps **one** slot per key token = whichever was written last, and the
leak-proof shuffle makes that 50/50. So hop2 = P(t1's slot is the real one) × P(retrieve)
= 0.5 × 0.9 ≈ 45%. hop1 is untouched (t0 is only ever a key). The chaining mechanism is
sound; a store artifact caps the second hop.

## Hypothesis
**Don't store pad-valued slots.** pad is not a fact, and the spurious `t1→pad` slot is
exactly what clobbers the intermediate. `--write-skip-pad` skips any write whose value
token is pad, so t1's only slot is `t1→t2` — removing the collision **order-independently**
(no reliance on which occurrence was written last).

## Predictions
- hop2 → ~hop1 (~98%); **2-hop joint → ~0.98² ≈ 96%** (up from 45%).
- **Broken-chain control at chance:** with `A→B, X→C` and query `A⇒C`, hop1 (A→B) still
  fires but hop2 has no `B→C` to find → joint ≈ 0%. This separates real chaining from any
  "return last value" leak.
- **Replicates across seeds**, and **extends to 3-hop** (~0.98³ ≈ 94%).

If 2-hop lands ~96% with the broken control at the floor, seed-replicated, and 3-hop
holds, that is **CoT-over-memory working** (the north star) via explicit emitted
intermediates — route A, one bounded shared store, facts held out-of-window.

## Runs (warm from r79_shared_gate_recall.pt; `--write-op-gate --write-skip-pad --recency-read --n-chains 4`)
- **box1:** `r84_chain` (n_hops 2, seed 0, 8k) + `r84_chain_broken` (broken-chain control, 3k).
- **box2:** `r84_chain_s1` (n_hops 2, seed 1, 8k) + `r84_chain_h3` (n_hops 3, seed 0, 8k).

Route B (latent loop-gate) was a flat negative (r82); route A is the viable route.
