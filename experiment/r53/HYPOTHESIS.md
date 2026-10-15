# r53 — can an UNSUPERVISED surprisal score keep the right slots (approximate the r52 oracle)?

r52 proved the r51 cliff is the eviction *policy*: an oracle that protects the 127 fact slots bridges
31 windows at 99% from a 256-slot store, where the ring buffer cliffs to 0%. But the oracle is *told*
which slots are facts. r53 asks whether a signal already in the model finds them **without labels**.

## The signal

**Surprisal** — the write-time `−log p(token)` under the model's own head. The fact tokens are random
single words (keys and values drawn from disjoint word pools), so given their prose/context they are
**unpredictable → high surprisal**; distractor prose is natural text → **lower surprisal**. So keeping
the highest-surprisal slots should keep the facts, no supervision required.

## Setup — offline, no training

`mech.persist evict` extended with a per-example `surprisal` policy (keep the top-M slots by
`−log p(token)` captured during the write pass). Three policies compared at fixed physical M:

    uv run python -m mech.persist evict --run r50_bpe64.json --ckpt r50_bpe64.pt \
        --policies ring oracle surprisal --mem-list 256 512 --nseg-list 4 8 16 32 --out r53_surprisal.json

- **box2 (main):** 64-fact checkpoint `r50_bpe64` (seed 1).
- **box1 (cross-check):** 16-fact checkpoint `r50_bpe16_s2` (seed 2) — does the surprisal signal hold
  across fact count and seed, or is it a 64-fact artifact?

## Hypothesis

**Claim:** surprisal-eviction tracks the oracle — ~90–99% out to 31 windows where ring is at 0% —
because the random-word fact tokens sit at the top of the surprisal ranking and survive pruning.

**Predict:** surprisal ≈ oracle within a few points; the gap (if any) is prose tokens that happen to be
high-surprisal (rare words, names) crowding out a fact slot, worse at small M where every slot counts.

**Falsified if** surprisal ≈ ring (collapses as n_seg grows) — then prose surprisal overlaps the facts
too much for a threshold to separate them, and an unsupervised keep needs a different signal (a trained
selective write gate, r50's gate being near-open) or the facts must be tagged at write time.

## What it decides
- surprisal ≈ oracle → the deployable "keep the right slots" is real and label-free; the recipe's 2nd
  axis is usable off the shelf, and effective context at fixed M is set by the working set, not M/128.
- surprisal ≪ oracle → keeping facts needs a learned/tagged signal; write up the gap as the next lever.
