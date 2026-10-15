# r52 — is the ring buffer, not capacity, what caps effective context? (fact-protecting eviction)

r51 measured a plain ring-buffer store bridging ≈ M/128 context-windows and then collapsing to **0%**
the instant prose laps segment 0's facts (hard cliff exactly at `slots == M`). The store threw away the
*valuable* facts to make room for *worthless* distractor prose, purely because the facts were written
first. This round asks whether that cliff is a property of the **eviction policy** rather than of
capacity — and measures the ceiling a fact-aware eviction could reach, to decide whether building one
(the r26–r33 consolidation path ported to BPE) is worth it.

## Setup — offline, no training

`mech.persist evict` on the trained 64-fact checkpoint (`r50_bpe64`). Every segment is written to a
large store (no eviction), then the store is **pruned to a fixed physical size M** under two policies
and the query is read from the pruned store:

- **ring** — keep the most recent M slots (= r51: seg-0 facts, written first, evicted first).
- **oracle** — protect the `F = seg_len − value_shift = 127` fact slots and evict oldest **prose**
  instead. This never drops a fact, so it is the upper bound on any fact-aware eviction.

    uv run python -m mech.persist evict --run r50_bpe64.json --ckpt r50_bpe64.pt \
        --mem-list 256 512 --nseg-list 4 8 16 32 --out r52_evict.json

## Hypothesis

**Claim:** oracle recall stays high across n_seg where ring cliffs to 0% — because the facts are never
evicted, and the only cost of more prose is read-time interference, not fact loss. So the r51 cap is the
**buffer policy**, and a fact-aware eviction bridges arbitrarily many windows at fixed M.

**Predict:** at M=512, ring reproduces r51 (98.8% at n_seg=4, then 0% at 8/16/32); oracle holds
~90–99% at every n_seg (a mild taper as prose interference grows, never the cliff). At M=256 (below
even one fact segment + margin) oracle still protects the 127 fact slots and stays well above ring.

**Falsified if** oracle also collapses as n_seg grows (then interference from a full store — not
eviction of the facts — is the real limit, and a fact-aware eviction would NOT help; the r51 cap would
be fundamental, and the write-up says so and closes this thread).

## What it decides

- Oracle ≫ ring → the cliff is policy, not capacity; **r53 builds the unsupervised version** (score
  slots by surprisal / a learned write gate so facts are kept without being told which they are), and
  the "how to store more" recipe gains a second axis: *keep the right slots*, not just *add slots*.
- Oracle ≈ ring → capacity/interference is fundamental at fixed M; the recipe stays "grow M" and this
  thread closes.
