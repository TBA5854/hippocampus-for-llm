# r3 — same-space keys: does the fact-count cliff disappear?

## Ideate (from r2, mid-round)

r2's layer-1-key run (`w5s0k1`) showed the cleanest split of failure modes so far: content-miss 3.1%,
so **whenever the read finds the right slot, the answer comes out**, but recall pinned at ~53% with 2
facts, because the read finds a *key position* without discriminating *which* key. Right place, wrong
key — the same ~1/n signature as the old 4-fact 28.3% and the 2-fact DG-on 50.9%.

The cause is structural. The stored key is the raw layer-1 state; the query is that state passed through
a randomly initialised projection. The key side cannot learn (it enters the store detached), so every
bit of alignment has to be found by the query projection from noise. With `query_init=identity`, query
and key are the same function of the same state at step 0. Smoke test: **correct-slot reads 90% at
step 10**, against 1-5% from a random start.

## Hypotheses

- **H3a — the cliff was matching.** With same-space keys, 8 facts lifts off within the same 4000-step
  budget that left the old architecture at chance. (Old: 8 facts 4.1% best.)
- **H3b — it reaches the objective's placement cleanly.** 2 facts reaches 100% with the value from the
  last block, which r1 could not.
- **H3c — r2's prediction pairing is real.** Rerunning `w5s1` with the offset diagnostic, reads
  concentrate at offset +1 (the slot one before the key), not 0.

## Falsifiers

- 8 facts stays near 1/8 with content-miss low -> matching still the limit; identity init helps at
  step 0 but training pulls the query away (would show as `align +0` decaying).
- 8 facts fails with **content**-miss high -> the value from the last block is the limit once matching
  works; next lever is the value source or shift.
- Scrambled control above chance -> the gain is capacity, not stored content.

## Runs (sequential, most decisive first; st_bg4 config otherwise)

Placement for all but the last: `--write-layer 5 --key-layer 1 --value-shift 0 --query-init identity`.

| order | name | facts | purpose |
|---|---|---|---|
| 1 | r3_f8 | 8 | H3a — the cliff |
| 2 | r3_f8_shuf | 8 | control for 1 |
| 3 | r3_f2 | 2 | H3b |
| 4 | r3_f16 | 16 | if the cliff moved, where is it now |
| 5 | r3_w5s1_align | 2 | H3c, r1/r2 placement (`--write-layer 5 --value-shift 1`) |
| 6 | r3_w4_align | 2 | instrument check at the right offset: the 100% reference must read at +1 |

**Added after r2 step 1000:** `r2_w4`, the 100% reference, fell to 0% identity-slot (offset 0) reads as
its recall climbed. So block 4 also pairs by prediction, and r1's diagnosis — "block 4 keeps identity,
block 5 loses it" — cannot be why block 5 collapses. The instrument check moves to offset +1.

Every run saves its checkpoint (`--save`) so later diagnosis does not need retraining.
