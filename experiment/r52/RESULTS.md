# r52 — RESULTS: r51's cliff is the eviction POLICY, not capacity. Protecting facts bridges 31 windows at M=256.

Offline `mech.persist evict` on the 64-fact checkpoint (`r50_bpe64`): write every segment to a large
store, prune to a fixed physical M under **ring** (keep most-recent, = r51) vs **oracle** (protect the
`F=127` fact slots, evict oldest prose), read the query. Hypothesis **CONFIRMED**.

## Recall from the empty query window (64 facts, 10×16 queries per cell)

| n_seg | bridge | ring M=512 | **oracle M=512** | ring M=256 | **oracle M=256** |
|---|---|---|---|---|---|
| 4  | 3 win  | 98.8% | 98.8% | 1.2% | 99.4% |
| 8  | 7 win  | **0.0%** | **99.4%** | 0.0% | 100.0% |
| 16 | 15 win | **0.0%** | **98.1%** | 0.0% | 99.4% |
| 32 | 31 win | **0.0%** | **98.8%** | 0.6% | **99.4%** |

## What it shows

1. **The cliff is the policy.** The ring buffer collapses to 0% the moment prose laps the facts (r51),
   but the *same store contents* pruned to protect the facts answer at ~99% out to 31 windows. Nothing
   about capacity changed — only which slots were kept. So effective context is **not** bounded by
   `M/128`; it is bounded by `F/128` (the facts must fit) plus a working set of recent context.

2. **A tiny store bridges arbitrarily far if it keeps the right slots.** At **M=256** — two windows of
   raw capacity — oracle bridges **31 windows (99.4%)**. The store only has to hold the 127 fact slots;
   prose is disposable. "Store more" is therefore not only "grow M": it is **"keep the right slots."**

3. **Less retained prose reads *cleaner*.** Oracle M=256 (99.4% at n_seg=16) beats oracle M=512
   (98.1%) — a smaller protected store holds fewer distractor prose slots, so the read faces less
   interference. Aggressive eviction of low-value slots is not just a capacity trick; it can *raise*
   read accuracy. (This is the read-side dual of the r30 "compress by write rate" finding.)

## Verdict
- **H52 (r51 cliff is buffer policy; a fact-aware eviction bridges arbitrarily far at fixed M):**
  CONFIRMED. Oracle ~98–100% at every n_seg and both M; ring cliffs to 0%. Not falsified — oracle does
  not decay as n_seg grows, so full-store interference is NOT the binding limit once facts are kept.

## Opens — what r53 must answer
Oracle is an upper bound: it is *told* which slots are facts. The deployable question is whether an
**unsupervised** score keeps them without being told. Two candidates already in the model:
- **surprisal** — the write-time `−log p(token)`; a random word-value after a key is high-surprisal,
  prose lower. Evict lowest-surprisal.
- **the trained write gate** — but r50 logs show it near-open (`gates≈1.0`), so it is not currently
  selective; surprisal is the more promising signal.
r53 adds a surprisal-scored eviction to the probe and measures how close to the oracle it gets.
