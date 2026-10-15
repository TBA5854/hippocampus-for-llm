# r30 — compression by WRITE RATE: how few slots does an episode need?

## Ideate (the user's observation)

"the day is good — by the time i come to good, the other 3 tok already have shit and that's accumulating, so maybe the
model writes once for an episode, or as few as possible, instead of every token."

That is right about the mechanism. With causal attention, the state at position t has already attended over everything
<= t, so one slot per token stores T overlapping summaries of the same episode. The redundancy compounds with length.

It also reframes what "compression" means here. The plan's original idea was a smaller `d_mem` (4x at best, and it
shrinks what each slot can hold). **Slots per episode is the bigger lever and it multiplies with the other one**: 1 in
8 tokens is 8x, before any change to slot width.

But the naive version is exactly what failed before. v0.1-v0.3 wrote one pooled summary per episode and could not carry
an exact binding; r9 explains why — token identity survives only early in the stack, which is why the value needs the
block-0 skip. A single end-of-episode vector is the thing that lost the facts. So the question is not "one write or
every write" but **which tokens are worth a slot**, which is the write gate deferred at the start of the project.

## Hypotheses

- **H30a — stride 2-4 costs little.** Recall stays >= 95% at half or a quarter of the slots. Falsified if recall tracks
  the slot count down, which would say every token really does carry distinct content.
- **H30b — a surprisal gate beats stride at equal rate.** Writing only where the model could NOT predict the token that
  slot would return is r6's finding inverted: a store holding what the model can already predict was useless, so do not
  spend slots on predictable tokens. A fact seen once is maximally surprising; the fourth token of "the day is good" is
  not. Falsified if the two are equal — then position is as good as content, and the cheap rule wins.
- **H30c — episode-end only fails.** One slot per segment, the v0.1 design. Expected at chance; run because a failure
  that reproduces a known failure is evidence the diagnosis was right, not a wasted run.
- **H30d — decode coverage degrades with rate.** r26 made sleep possible by showing the store is an exact transcript.
  Skipping tokens puts holes in it. Recall can survive holes (the facts are what matter); **replay cannot** if the
  holes fall on fact tokens. Reported separately, because this is where write-rate compression could quietly break
  consolidation.

## A diagnostic that stops being valid (noticed at step 250)

`slot@1`, `mass` and the alignment columns all assume **slot p holds position p** — the mapping `write_many` gives when
every token is written. Under selection that mapping is gone, so those columns are meaningless here and must not be
read as matching failures. First evals show it plainly: stride 2 reports `slot@1 13.1%, match-miss 86.7%` while
surprisal reports `slot@1 100%` — the second only because early in training the scores are near-uniform and `topk`
breaks ties toward the lowest indices, which is exactly where the facts sit. Neither number means what it usually
means.

**Recall is the honest metric for this round.** The slot diagnostic would need the selected indices carried into the
store to be repaired, which is a change to make only if this round justifies it.

**Correction to an earlier note in this file.** I claimed stride skips fact keys in proportion to k. That is wrong for
k=2: the untiled layout writes pairs as `k v k v ...` from position 0, and slot i pairs position i with i+1, so fact
slots are exactly the EVEN i — precisely what stride 2 keeps. Stride 2 retains **all eight facts**.

It matters because it changes what the stride-vs-surprisal gap measures. At k=2 both stores hold every fact at the same
slot count, so the gap (stride 10.8% at step 4750 vs surprisal 53.9% at 3500) is about **composition**, not coverage:
the surprisal store is filled with informative slots, the stride store spends half its slots on predictable filler.
That is r5's correlated-key failure in a new guise — near-duplicate keys leave no signal to retrieve on.

Coverage does bite at higher k: stride 4 keeps fact keys 0,4,8,12 (half the facts) and stride 8 keeps 0,8 (a quarter),
so those two runs mix coverage with composition and must be reported as mixing them.

## The measurement that matters

Recall per slot, not recall. A run at 25% rate that holds 95% recall is a 4x win; the same recall at 100% rate is not.
Reported as slots/example alongside recall, plus decode coverage on fact symbols specifically.

## Runs (1x store, 128-symbol vocabulary, 8 facts, cold, 8000 steps — r23's configuration, the one that trains from
## scratch, so no warm start confounds the comparison)

| box | name | rate |
|---|---|---|
| box2 | r30_stride2 / r30_stride4 / r30_stride8 | every 2nd / 4th / 8th position |
| box1 | r30_surp50 / r30_surp25 / r30_surp12 | top 50% / 25% / 12.5% by surprisal |
| box1 | r30_last | one slot per segment (the v0.1 control) |

The stride and surprisal arms are matched in rate (1/2 vs 50%, 1/4 vs 25%, 1/8 vs 12.5%), which is what makes H30b a
fair test.
