# r59 — real breaking-point sweep (128 symbols = r24 cliff regime)

r57/r58 used the 32-symbol default (easy regime): everything <=32 facts hit 100%, f64 crashed (>vocab).
That could not locate the cliff. Here n_symbols=128 so fact count is a small fraction of the key vocab
and interference bites (r24: single-layer 8f 99.7 -> 16f 3.8).

## Hypothesis (user): the unique per-layer store holds LONGER than the shared single-layer store.
Per-layer arm: pl @ 8/16/24/32. Compare vs single-layer baseline (sl) and shared symmetric (sh) @ 16/32.
If pl holds at 16/24/32 where sl cliffs to ~4%, the per-depth memory carries bindings the single
layer-0 key cannot -> reopens where memory lives, and is the anti-CSA2 datapoint.
