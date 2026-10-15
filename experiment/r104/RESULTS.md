# r104 — per-layer write PUSHES the 128-fact capacity wall (0.3% → 45.3%), but does not eliminate it

**The r88 capacity frontier belongs to the single mean-write store, not to the architecture.
At 128 distinct facts the mean-write store (one averaged value per token) never bootstraps
(0.3%, flat to 10k steps), but PER-LAYER write (each layer stores its own value into the
one shared store, exact-token-keyed) reaches 45.3% — a 150× lift, genuinely learning where
the single value could not. It is a partial win: 128 facts still fall well short of the
~95% per-layer reaches at 32/64, so per-layer write PUSHES the frontier substantially
rather than eliminating it. The gain rides on the deep read gates opening (layers 3–5 to
0.51–0.61) — the extra per-depth slots are what give 128 facts a foothold.**

## Numbers (cold bpe_recall, d256/6L, n_seg 8, 10k steps, best = final)
| facts | arm | slots/token | recall (best = final) |
|-------|-----|-------------|-----------------------|
| 64  | baseline_mean | 1 | **96.3%** |
| 128 | baseline_mean | 1 | **0.3%** (flatlined, never bootstrapped) |
| 128 | **spl_token** (per-layer write) | 6 | **45.3%** |

(The 128-fact spl arm ran at batch 16 to fit the store on a 16GB GPU; the mean-write
baseline is batch 32. The 0.3% vs 45.3% gap dwarfs any batch effect. Trajectory:
per-layer bootstraps at step ~4k and climbs 5→12→24→35→43→45%, decelerating to a 45.3%
plateau by 10k; the mean-write baseline sits at 0.3% throughout.)

## Reading
- **The wall was the single mean-write value, not the architecture.** r88 found 128 facts a
  hard frontier for the bounded store; that store averaged all layers into one value per
  token. Giving each layer its own slot (per-layer write, r103) lets 128 facts bootstrap to
  45.3% where the mean-write store is pinned at chance — so the frontier is a property of
  how much distinct content one token's slot(s) can carry, and per-layer write raises it by
  storing L views instead of one.
- **Still partial at 128.** Per-layer write reaches ~95% at 32/64 facts (r103, and the r104
  64-fact baseline is already 96.3%) but only 45.3% at 128, and the climb had flattened by
  10k (45.3% at both 9.5k and 10k) — so this is a real ceiling for this configuration, not
  just under-training. 128 distinct facts remain hard; per-layer write moves the wall, it
  does not remove it.
- **The deep gates carry it.** The mid/deep read gates open to 0.51–0.61 (layers 3–5) as
  recall climbs, versus the near-shut layer-0 gate (0.03) — the extra per-depth slots at
  depth, not the layer-0 token slot, are what give the 128-fact regime a foothold.

## Verdict
Per-layer read/write (the r103 default) **substantially pushes the 128-fact capacity
frontier — 0.3% → 45.3%** — turning a hard wall for the mean-write store into partial recall,
but does not fully close 128 facts (partial vs ~95% at 64). So limitation "capacity has a
real frontier" is revised: the frontier is pushed by per-layer write, not eliminated; fully
closing 128 facts likely needs more capacity still (more heads, a larger d\_mem, or several
shared stores — the r105 direction). Folded into paper §limitations. This completes the
per-layer read/write arc (r103–r104).
