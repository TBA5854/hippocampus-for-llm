# r90 — curriculum FIXES the gate×depth cold-start

**Warming the write-op gate from the r87 plain-store checkpoints turns r85's cold gate
failure into full recall: d256/6L 0.3%→94.8%, d192/12L 0.2%→95.6%, both bounded to
O(facts) (mem ~228). So the gate×depth cold-start (r85) is a curriculum artifact, not a
capability limit — exactly like the fact-count cliff (r25/r66) was under-training.**

## Numbers (bpe_recall, 32 facts, M=1024, +write-op-gate, warm from r87 plain ckpt, 6k)
| shape | r90 gate warm | r85 gate COLD | r87 plain (warm source) | mem |
|-------|---------------|---------------|-------------------------|-----|
| d256 / 6L  | **94.8%** | 0.3% | 84.7% | 228 |
| d192 / 12L | **95.6%** (best 95.8) | 0.2% | 97.0% | ~228 |

## Reading
- Both shapes that could NOT cold-bootstrap the gate (r85: 0.3%, 0.2%) reach **~95%** when
  the gate is warmed from a converged plain-store model. The read path is already
  established by the plain warm-source, so turning the gate on only has to learn the
  new/update/skip routing, which it does — and the store bounds to O(facts) (mem 228 vs the
  plain store's 508), the whole point of the gate.
- d256/6L ends slightly above its plain warm-source (84.7→94.8): the gate run also trains
  longer on the recall task, closing the gap the plain 6-layer had.

## Verdict
Completes the r85→r87→r90 arc: r85 found the gated store fails cold in a deep stack; r87
showed the plain store is depth-robust (so it's the gate, not depth); r90 shows a
**curriculum (warm the gate from the plain store) fixes it**. The managed bounded store is
therefore not depth-limited — it just needs the plain store bootstrapped first, one warm
rung, the same recipe that cleared every earlier cliff. Folded into the paper limitations.
