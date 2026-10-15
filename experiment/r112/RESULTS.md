# r112 — RESULTS: Passkey (needle-in-a-haystack) — no lost-in-the-middle, but a write-order decay

**Verdict: the content-addressed store has NO lost-in-the-middle, but recall decays
strongly with the needle's WRITE-ORDER (depth) — a fact stated early persists, a fact
stated late into a saturated store is lost. Cross-model, controls at chance.** This
overturns the naive "flat across depth" hypothesis with a sharper, honest result.

## Setup
`mech.persist passkey` (new probe): a fact (the passkey) is planted in ONE of `ctx`
filler windows at depth `d`, queried from a separate window; top-1 read; batch 2.
`needle_seg` places the fact (default 0 = historical seg-0; RNG byte-identical). Sweep
depth = fraction of context. Two models: d512/2-layer (r87), d256/6-layer (r92).

## Results — recall vs needle depth

### d512 (strong model)
| ctx (tokens) | d0.0 | d0.25 | d0.5 | d0.75 | d1.0 |
|---|---|---|---|---|---|
| 64 (~8k)  | 100% | 100% | 100% | 87.5% | 75% |
| 512 (~64k) | 100% | 62.5% | 50% | 25% | 12.5% |

### d256 (weak model)
| ctx (tokens) | d0.0 | d0.25 | d0.5 | d0.75 | d1.0 |
|---|---|---|---|---|---|
| 64 (~8k)  | 87.5% | 0% | 0% | 0% | 0% |
| 512 (~64k) | 87.5% | 0% | 0% | 0% | 0% |

### Control (d256, ctx64, `--shuffle`): **0% at every depth** — retrieval is genuine.

## Reading it
- **No lost-in-the-middle.** The classic attention failure is a *dip in the middle*;
  here recall is **monotonic from depth 0** — the middle is never anomalously bad. On
  d512 the mid-context needle (depth 0.5) recalls at 100% (8k) / 50% (64k).
- **Strong write-order / primacy effect.** A fact written **early** (depth 0) persists
  (100% / 87.5%); a fact written **late into a saturated store** is lost (12.5% on
  d512, 0% on d256). The store privileges what it saw first — the *opposite* of a KV
  cache's recency.
- **Cross-model, graded by capacity.** The decay is graded on the strong model and
  **near-binary on the weak one** (only the first-written fact survives). So it scales
  with model/store capacity — connecting directly to the 128-fact capacity frontier,
  now seen along the *time* axis.
- **Scales with saturation** (mild at 8k, severe at 64k) — characterized in r113.

## Mechanism → r113
r113 decomposes the cause (saturation vs novelty write-gate vs ring lapping vs
key-centering). Preview: lapping ruled out (grow == baseline); the novelty write-gate
is *part* of it (rare unique keys lift the late tail), with a residual second factor.

## Infra
Added `needle_seg` to `bpe_recall_batch` (default 0 = no-op), threaded through
`get_batch`, and a `passkey` probe in `persist.py` (+ `--depth-list`). Committed here.
