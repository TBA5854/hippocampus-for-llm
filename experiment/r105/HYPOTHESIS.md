# r105 — does MORE CAPACITY close the 128-fact gap that per-layer write left open?

## Context
r104: per-layer write PUSHES the 128-fact wall (mean-write 0.3% → per-layer 45.3%) but does
NOT close it — 45.3% is partial vs ~95% at 32/64, and it plateaus. Paper limitation #2 names
the open direction: "whether more heads, a larger d_mem, or several shared stores close the
remaining gap." r103/r104 used --n-heads 1. This tests the cheapest capacity lever: MORE HEADS.

## Hypothesis
More attention heads (4 vs 1) give the model more retrieval/representation capacity, so
per-layer write at 128 facts reaches higher than r104's 45.3%. If recall rises materially
(say >60%), capacity is the lever and the frontier keeps moving with model size. If it stays
~45%, heads are not the bottleneck (points to d_model width or several stores instead).

## Design (cold bpe_recall, 128 facts, per-layer write, matched to r104 except heads)
- box2: r105_spl_token_f128_h4 — `--n-heads 4` (vs r104's 1), else identical: d256/6L, n_seg 8,
  batch 16, --shared-per-layer --shared-pl-key token, --store-grow, 10k steps, seed 0.
- Compare directly to r104 spl_token_f128 (n_heads 1) = 45.3%.
- (box1 is the user's; single-arm round on box2 only.)

## Predictions
1. If heads help: recall > 45.3% at 128 facts (capacity is the lever; frontier scales with size).
2. If not: recall ~45% (heads are not the bottleneck; the gap is width or store count).
Either is a clean answer to limitation #2's open direction; a null is written up, not re-tuned.

## Controls
- The r104 n_heads-1 result (45.3%) is the matched baseline (same everything but heads).
- shuffle-store control at eval if the result is ambiguous.
