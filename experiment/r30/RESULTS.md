# r30 — results: 4x compression is free, if the gate picks by content

## Numbers (1x store, 128 symbols, 8 facts, cold, 8000 steps — r23's configuration; chance 0.8%)

| write policy | rate | slots/segment | fact coverage | **final recall** | gate (final) |
|---|---|---|---|---|---|
| all tokens (r23 baseline) | 100% | 381 | 100% | 99.7% | 0.16 |
| **surprisal** | 50% | 192 | 100% | **99.7%** | 0.277 |
| **surprisal** | 25% | 96 | 100% | **100.0%** | 0.163 |
| surprisal | 12.5% | 48 | 90.6% | 3.6% | 0.038 (stalled) |
| stride 2 | 50% | 190 | 100% | 58.0% | 0.094 |
| stride 4 | 25% | 95 | ~50% | 0.5% | 0.003 (shut) |
| stride 8 | 12.5% | 47 | ~25% | 0.5% | 0.003 (shut) |
| episode-end only | 1 slot/segment | 3 total | — | **0.5%** (chance) | 0.003 |

## Verdicts

- **H30a (stride 2-4 costs little) — falsified.** Stride 2 ends at 58.0% and stride 4 at chance, against a 99.7%
  baseline. Position-based skipping is not cheap; it is close to fatal.
- **H30b (surprisal beats stride at equal rate) — confirmed, and by more than expected.** At 50%: 99.7% vs 58.0%.
  At 25%: 100.0% vs 0.5%. Same slot budget, same everything else — only *which* tokens earn a slot differs.
- **H30c (episode-end only fails) — confirmed** (chance throughout, gate 0.003), reproducing the v0.1-v0.3 pooled-
  summary failure that r9 diagnosed: one vector per episode cannot carry an exact binding.
- **H30d (decode coverage degrades with rate) — falsified where it matters.** Coverage of fact keys stays at **100%**
  at both working rates, and 90.6% even at the rate that fails. **Replay for sleep survives compression** — the holes
  fall on predictable text, which is exactly what the gate is supposed to drop.

## Diagnosis

**The user's premise was right and the mechanism is sharper than "redundancy".** With causal attention, position t's
state already summarises everything up to t, so one slot per token stores the episode many times over. But the win does
not come from writing *fewer* slots — stride writes exactly as few and collapses. It comes from writing slots that are
not near-duplicates of each other. That is r5's correlated-key failure seen from the write side: predictable tokens
produce keys that crowd the same region, and a store full of them has no signal to retrieve on. Surprisal filters them
out by construction.

The clearest single comparison: at 25% both policies write ~96 slots; surprisal reaches **100%**, stride reaches
**0.5%**. Nothing separates them but content-awareness.

**Compression also accelerates learning.** Surprisal at 50% passed 50% recall at step ~3500 and 99% by ~6750; the
full-write baseline needed the entire 8000 steps to reach 99.7%. Fewer, better slots make the read easier to discover,
not harder — the opposite of the natural expectation.

**Where it stops, and why it is not capacity.** 12.5% (48 slots) fails at 3.6% — but its fact coverage is 90.6%, so
the facts are *in the store*. The gate stalls at 0.038 and never opens. This is the bootstrapping deadlock of r25
again: too few successful early reads to pay for opening the gate. It is a lift-off failure at a viable store, which
predicts a curriculum would fix it (write at 25%, then tighten) — untested, and the obvious follow-up.

## Compression achieved

**4x, free.** 96 slots per segment instead of 381, at 100.0% vs 99.7%. This multiplies with `d_mem` compression, which
is untouched here and is the *other* axis the plan originally meant: at `d_mem=64` the combined saving would be 16x,
and nothing in these results speaks against it.

The surprisal rule is also cheap to compute: the model's own next-token NLL, already available in the forward pass, no
extra parameters and no second model.

## What this does NOT show

- One seed, one configuration (1x store, 8 facts, 128 symbols, cold training).
- The gate is an **oracle over teacher-forced text**: it scores positions by the NLL of the token that actually
  follows. That is available during training and during a wake pass over text already seen — but a deployed writer
  choosing in real time has the same information (it has just read the token), so this is a fair rule, not a cheat.
  Worth stating explicitly because it would be a cheat in a generation setting.
- Slot-level diagnostics (`slot@1`, `mass`, alignment) are **invalid under selection** — they assume slot i holds
  position i. Recall and coverage are the honest metrics here; the logged slot columns should be ignored for this
  round.
- Interaction with sleep is untested end to end: coverage says replay *should* survive, but no consolidation run has
  been done on a compressed store.

## Follow-up, run the same day: a RATE curriculum reaches 8x

r30 predicted this. 12.5% failed cold at 3.6% with fact coverage still at 90.6% and the gate stalled at 0.038 — a
lift-off failure at a viable store, which is exactly what r25 showed a curriculum escapes. Warm-started from the 25%
checkpoint:

| step | 250 | 500 | 1000 | 1250 | 2000 |
|---|---|---|---|---|---|
| recall | **73.4%** | 69.4% | 82.0% | 88.9% | 88.4% |
| gate | 0.166 | 0.187 | 0.229 | 0.241 | 0.269 |

Against 3.6% for the whole 8000 steps cold. **8x compression (48 slots/segment) works once bootstrapped**, and the
matched scrambled control at the same rate is running on box2. 6.25% (24 slots) follows in the same curriculum.

So write-rate behaves exactly like store size (r19-r22), fact count (r25) and vocabulary (r23): the cold failure is the
gate/decoder bootstrapping deadlock, and a curriculum escapes it. That is now **four axes, one failure mode**.

## Seed 1: the compressed run replicates, and the BASELINE does not

Everything since r19 is seed 0, so the headline claim got a second seed — with its own matched full-write baseline:

| write policy | seed 0 | seed 1 |
|---|---|---|
| **25% surprisal** | **100.0%** | **100.0%** |
| 100% (write everything) | 99.7% | **27.8%** |

The compressed configuration is identical across seeds. The full-write baseline collapses to 27.8% on seed 1 — with
`slot@1` at 100% and read mass 1.00 throughout, so it is a content/lift-off failure, not a matching one: it was still
climbing at step 8000 (4.7% at 5250, 11.1% at 6000, 18.4% at 6250, 27.8% at 8000), i.e. **the full store lifts off so
late that 8000 steps is not enough on this seed**.

That inverts the usual reading of these numbers. Compression is not a concession that costs a little accuracy for a
lot of memory; on this evidence the redundant store is the *fragile* configuration, and removing near-duplicate slots
is what makes the read learnable at all. It is consistent with the r5 correlated-key failure and with r30's own
finding that compression accelerates learning — but it is a stronger claim than the round originally made, on two
seeds rather than one.

## Next

- r31 (episodes) can now assume a compressed store: 4-8x fewer slots per episode changes what "a day's worth" means.
- **Slot-width compression: answered, negatively** — see r31d. It fails at chance for d_mem 64 and 32, because a
  projection destroys the exact-token matching that layer-0 keys provide. The two compression axes are not
  interchangeable; only write rate is available in this architecture.
