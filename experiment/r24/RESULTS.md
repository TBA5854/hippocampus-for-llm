# r24 — results: the cold fact-count cliff is between 8 and 16, and it is graded

## Numbers (1x store, 128-symbol vocabulary, cold, 8000 steps, box2; r18's configuration, seed 0)

| facts | final | best | read mass (start -> end) | gate (end) | slot@1 |
|---|---|---|---|---|---|
| 8 (r23) | **99.7%** | 99.7% | 0.98 -> **1.00** | opens fully | 100% |
| 16 | 3.8% | 5.2% | 1.00 -> 0.99 | 0.057, frozen from ~3000 | 100% |
| 24 | 3.6% | 4.7% | 1.00 -> **1.00** (dips to 0.88, recovers) | 0.038, frozen | 100% |
| 32 | 3.1% | 3.6% | 1.00 -> 0.97 | 0.024, closed to 0.007 then partly reopened | 100% |
| 48 (r23) | 0.9% | 1.6% | 0.97 -> **0.70** (decays throughout) | 0.008, shut | 100% |

Chance 0.8%. Note the recall column for 16-32 facts sits at 3-4%, *above* the 0.8% chance line but far below use:
with 128 values the model can still favour the few value symbols it has seen recently.

## Verdict

**The cold cliff is between 8 and 16 facts, and beyond it the failure deepens gradually rather than switching off.**
Read mass tells the cleanest story: 8 facts concentrates to 1.00, 16-24 facts holds ~1.00 but never converts it, 32
facts erodes to 0.97, and 48 facts dilutes to 0.70.

Every run finds the correct slot 100% of the time, at every eval, at every fact count. Not one of these is a matching
failure.

## Diagnosis

Two distinct regimes, and only the second looks like interference:

1. **16-24 facts — a stalled start.** The read is perfectly concentrated (mass ~1.00) and the gate opens a little and
   then freezes (0.038-0.057). The model has a usable read and does not learn to use it.
2. **32-48 facts — dilution.** Mass decays monotonically and the gate shuts. This is the r16/r17 signature: without
   early successful reads there is no pressure keeping key representations apart.

What this round does *not* show is that any of these are capacity limits. Against that reading: 508 slots hold 48 facts
with room to spare, and **32 facts already reached 100% at the 8x store when warm-started** (r21). The store-size axis
had exactly this shape — cold runs stuck at chance, warm starts at 100% — and the fix there was a curriculum.

## Next

r25 (running): a curriculum over *fact count* rather than store size — 8 -> 16 -> 48 facts at a fixed 128-symbol
vocabulary, each warm-started from the previous checkpoint, with a matched scrambled control at 48. If that works, the
fact count was never a capacity limit either, and this project has found one failure mode wearing several costumes.
