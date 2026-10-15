# r62 — RESULTS: the north-star result is bulletproof — O(1) slots per query, control at chance, scales to 63×

Hardens r61 with the three things it lacked: a matched scrambled control, a second model, and a bigger
store. Offline `mech.persist sparse` (top-k read of a store bridging K context windows).

## Real reads — top-k vs dense (recall)

**r50_bpe64 (64 facts):**

| store | K windows | slots | **top-1** | top-2 | dense (all) |
|---|---|---|---|---|---|
| M=2048 | 15 | 1905 | **100.0%** | 96.9% | 93.8% |
| M=4096 | 31 | 3937 | **100.0%** | 95.0% | 90.0% |
| M=8192 | 63 | 8001 | **100.0%** | 90.6% | 82.5% |

**r50_bpe16_s2 (16 facts, cross-check):**

| store | K windows | top-1 | **top-2** | dense (all) |
|---|---|---|---|---|
| M=2048 | 15 | 78.8% | **100.0%** | 99.4% |
| M=4096 | 31 | 80.6% | **100.0%** | 99.4% |

## Matched control (shuffle: read a neighbour's store)

| model @ 31 windows | top-1 | dense |
|---|---|---|
| r50_bpe64 shuffle | **0.0%** | 0.0% |
| r50_bpe16_s2 shuffle | **0.6%** | 0.0% |

## What it shows

1. **The result is real.** With the store's contents permuted, recall is chance (0.0 / 0.6%) at every k
   — the top-k read genuinely finds the answer slot, it is not an artifact of the readout.

2. **O(1) slots per query, at any store size.** The number of slots a query needs is **1–2**, not a
   function of the store: bpe64 nails it at **top-1** (1 of 8001 slots at 63 windows); bpe16_s2 needs
   **top-2** (its answer binding spans two slots) and then hits 100%. Either way each query touches at
   most two of thousands of slots — the read is O(1) in store size.

3. **Small-k sparse ≥ dense, and the gap grows with the store.** On bpe64 the dense scan *decays* as the
   store grows (93.8 → 90.0 → 82.5 over 15 → 31 → 63 windows) because distractor mass accumulates in the
   softmax; top-1 is flat at 100%. On bpe16_s2 top-2 (100%) also edges dense (99.4%). Sparse retrieval is
   not a cost to pay for scale — it is what *makes* the store scale cleanly.

4. **The right k is small but model-dependent.** bpe64 peaks at k=1 (larger k pulls in a distractor at
   huge stores: top-2 90.6% at 63 win); bpe16_s2 peaks at k=2. A fixed small k (≈2) is a safe default;
   tuning it per model recovers the last point.

## Verdict
- **North-star (store ≫ window, query only what's required), hardened:** CONFIRMED. A store up to **63×
  the context window** answered from an empty prompt while each query touches **1–2 of up to 8001
  slots**, small-k sparse matching or beating the dense scan, with the scrambled control at chance on
  two independent models.

## Opens
- The score ranking is still a dense q·K scan; a sparse *candidate pool* (CSA2, route 2) is how to make
  the O(1)-slot read also O(1)-compute at very large M. [[csa2-store-future-build]]
- The per-model optimal k (1 vs 2) traces to how many slots carry the answer binding — worth a look when
  mech-interp becomes the focus, not now.
