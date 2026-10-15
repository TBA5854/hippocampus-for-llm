# r112 — Passkey retrieval (needle-in-a-haystack): depth-invariance

## The benchmark
Passkey / needle-in-a-haystack is the standard long-context probe: a "passkey" (a
fact) is planted **once** in a long context of filler and queried at the end. Two axes
matter — the context **length** and the needle's **depth** (where in the context it
sits). Attention transformers famously suffer **"lost in the middle"**: recall dips
for needles buried mid-context, because retrieval competes with position-biased
attention.

We already own the length axis (r102/r108: top-1 holds **100%** out to **1M tokens**).
r112 tests the axis we have not isolated: **depth**.

## Hypothesis
Our store is **content-addressed** by a position-free layer-0 token key — the read
does not depend on where in the stream the fact was written. So we predict
**recall is invariant to needle depth**: no lost-in-the-middle. The fact recalls
equally from the first window, the middle, or the last, at every context length —
on **both** models (d512/r87, d256/r92).

## Design
- **Probe** `mech.persist passkey`: fill a store across `ctx = n_seg-1` filler
  windows with the needle planted in window `d = frac·(ctx-1)`; query at the end;
  **top-1** read. Sweep `--depth-list 0 .25 .5 .75 1.0`.
- **Lengths**: `ctx=64` (~8k tok) and `ctx=512` (~64k tok), each on both boxes/models.
- **Control**: `--shuffle` (read a neighbour's scrambled store) at `ctx=64`, all
  depths → must collapse to **chance** (~0%), proving the recall is genuine and not
  a filler artifact.
- **batch 2** (valid for the shuffle control — batch-1 shuffle is degenerate, r111).

## Reuse / risk
`bpe_recall_batch` gains a `needle_seg` arg (default 0 = byte-identical to every prior
round; RNG order preserved). The fill/query/top-1 machinery is the exact, proven
recall path (r101/r102/r108). A 1-iter **smoke** runs before the full sweep; on any
error the round aborts in seconds. Fast recall evals (~minutes/cell), no training.

## Pass criteria
1. top-1 recall **flat across all depths** at each length (the headline: no
   lost-in-the-middle), on both models.
2. shuffle control at **chance** at every depth.
3. depth-invariance holds at both 8k and 64k (length-independent).

A dip at mid-depth would falsify the position-free-addressing claim — written up as-is
if so (negative results are not re-tuned).
