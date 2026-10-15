# r54 — how much store headroom does surprisal need to REACH the oracle?

r53 found label-free surprisal-eviction tracks the oracle in the easy regime but collapses at long
bridges / tight M, because PG-19 prose contributes its own high-surprisal tokens that crowd out the
~127 fact slots once enough prose accumulates. That gives a prediction: surprisal should reach the
oracle **once M is large enough to hold the fact slots PLUS the high-surprisal prose tail** at a given
bridge. This round measures that margin as a curve, turning r53's qualitative "needs comfortable M"
into a number.

## Setup — offline, no new code

`mech.persist evict`, policies `ring oracle surprisal`, fixed bridges (n_seg = 16 and 32), swept over a
fine M grid. box2 = 64-fact `r50_bpe64`; box1 = 16-fact `r50_bpe16_s2`.

    uv run python -m mech.persist evict --run ... --ckpt ... \
        --policies ring oracle surprisal --nseg-list 16 32 \
        --mem-list 256 384 512 768 1024 1536 2048 --out r54_<tag>.json

## Hypothesis

**Claim:** surprisal recall rises monotonically with M and meets the oracle once M exceeds a threshold
`M*(bridge, n_facts)` — roughly `F + (high-surprisal prose slots at that bridge)`. Longer bridge ⇒
larger M* (more prose tail); more facts ⇒ smaller gap (facts dominate the ranking head, r53).

**Predict:** at n_seg=16, 64f reaches oracle by M≈512–768; 32-window bridge needs more. 16f needs more
M than 64f at the same bridge (its ranking head is more contestable). ring stays ~0 throughout (its M*
is `total`, off the grid).

**Falsified if** surprisal is flat in M (then the failure is not a capacity-margin effect and r53's
mechanism is wrong) or never meets the oracle even at M=2048 (then prose surprisal overlaps the facts
irreducibly, and no amount of headroom fixes an unsupervised threshold — the learned-gate route is
mandatory, not merely better).

## What it decides
A clean `M*` curve → the deployable rule: "for a bridge of K windows and N facts, a surprisal-evicted
store of size ≥ M*(K,N) matches the oracle." That is the concrete, usable form of the r52 "keep the
right slots" recipe. If instead M* diverges, the write-up says surprisal-eviction is capacity-hungry
and points to the trained selective gate (r55).
