# r88 — capacity curriculum: the 64→128-fact wall is REAL, not under-training

**Warming the 128-fact store from the CONVERGED 64-fact model (a curriculum rung) with a
longer budget does NOT break the wall: it peaks at 47.2% then COLLAPSES to 3.1%. So the
64→128 frontier (r85) is a genuine capacity/stability limit of the single shared managed
store, not a bootstrapping/under-training artifact.**

## Numbers (bpe_recall, 128 facts, M=4096, warm from store_f64, 16k)
| run | best | final |
|-----|------|-------|
| r88 f128_from_f64 (curriculum rung, 16k) | **47.2%** | **3.1%** |
| r85 f128_from_r79 (direct, 8k)           | 46.1% | 26.1% |
| r85 store_f64 (64 facts) — for contrast   | 96.1% | 95.9% |

## Reading
- Neither warm path converges: from the converged 64-fact model it reaches **47.2% at best**
  then destabilises to **3.1%**; from the 16-fact r79 it topped out ~26%. The stronger
  curriculum (warm from f64) did **not** help — it peaked similarly and then collapsed
  further. So the wall is not under-training: a model that already manages 64 facts, given
  16k more steps, still cannot stably hold 128.
- The collapse (47%→3%) is the tell: the single shared managed store can partially bind
  128 facts but not **stably** — the gate/recency dynamics at 128 distinct keys in one
  M=4096 store are unstable, so training drifts off the peak. Contrast the flat 95.9% at
  64 facts. A real capacity/stability frontier sits between 64 and 128 for this one store.

## Verdict
Confirms r85's frontier as genuine. This is the opposite conclusion to every earlier
scaling "cliff" (r25/r66), which were all under-training — here more training and a better
warm start do not help, and the run actively destabilises. Directions that might move it
(not chased here): more heads / larger d_mem, multiple shared stores, or the fine-key
separation (r89) to reduce key interference at scale. Folded into the paper's limitations.
