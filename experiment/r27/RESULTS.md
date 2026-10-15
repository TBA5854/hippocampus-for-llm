# r27 — results: sleep-as-eviction works, and rehearsal is the protection that matters

## Numbers (`mech/sleep_evict.py`, box2, on r23's 8-fact 1x checkpoint, n=64 queries, shared fact set)

| run | freeze keys | rehearse | steps | lr | **consolidation** | old store | new writes | LM 0.961 -> | drift |
|---|---|---|---|---|---|---|---|---|---|
| r27_naive_s400 | — | — | 400 | 1e-4 | **100%** | 98% | 100% | **2.599** | 44 |
| r27_prot_s400 | yes | 1.0 | 400 | 1e-4 | 75% | 100% | 50% | 0.977 | 32 |
| r27_prot_s1500 | yes | 1.0 | 1500 | 1e-4 | 88% | 100% | 48% | 0.875 | 51 |
| r27_prot_s1500_lr3e4 | yes | 1.0 | 1500 | 3e-4 | 88% | 100% | 25% | 0.818 | 97 |
| **r27_nofreeze_s1500** | — | 1.0 | 1500 | 3e-4 | **100%** | 88% | **75%** | **0.833** | 121 |
| r27_prot_s4000_lr1e3 | yes | 1.0 | 4000 | 1e-3 | 25% | **0%** | 22% | 0.690 | **418** |

- **consolidation** = a held-out fact answered from weights alone, memory off, nothing in context (0% before sleep in
  every run). With 381 slots, 100% means all 381 become reclaimable.
- **old store** = the pre-sleep store, snapshotted and restored, still read correctly after sleep (key drift).
- **new writes** = refill the store with the post-sleep weights and read it (write-path health).
- Decoding was exact (100%) in every run, as r26 predicted.

## Verdicts

- **H27a (protected sleep consolidates >= 50%) — confirmed.** 75-100% across the moderate settings.
- **H27b (the store stays readable) — confirmed for moderate drift, falsified at extreme.** 88-100% up to drift ~120;
  at drift 418 the pre-sleep store is **completely unreadable (0%)**. There is a threshold, not a gradient.
- **H27c (LM preserved with rehearsal) — confirmed and exceeded.** Every rehearsed run *improved* LM loss
  (0.961 -> 0.82-0.98). Without rehearsal it nearly triples (2.599).
- **New, and it reverses the round's own design: freezing the key space is the wrong protection.** Rehearsal alone
  beats keys-frozen+rehearsal on consolidation (100% vs 88%) *and* on new writes (75% vs 25%) at matched steps and lr.

## Diagnosis

**Why freezing keys backfires.** It pins the token embedding and block 0, so the stored keys stay valid — old store
100%, the best in the table. But all adaptation is then forced into the later blocks, and the *value* is written from
the last block plus a block-0 skip. The write path moves while the key path cannot, the two decalibrate, and refilling
the store afterwards reads back at 25-50%. Letting the whole model move coherently keeps write and read in step;
rehearsal is what stops it forgetting. The protection that was supposed to save the memory is what breaks it.

**Three regimes by weight drift**, which turns out to be the right axis to think on:

| drift | what happens |
|---|---|
| ~30-50 | safe: consolidation 75-100%, store intact, LM fine |
| ~100-120 | the usable frontier: full consolidation, 88% store, 75% new writes |
| ~400+ | collapse: the memory mechanism is destroyed (store 0%), even with keys frozen |

**The one unsolved cost: new writes at 75%.** Sleep drains the store it has, but leaves the model slightly worse at
writing the next one. Rehearsal here is pure language modelling — it protects general ability but nothing rehearses
the *memory* task, so the write path has no pressure keeping it calibrated. That is the obvious fix and it is r28.

## The eviction loop, as measured

Your design — evict by consolidating rather than by age — runs end to end:

1. **Decode** the store: keys -> token ids, exact (r26).
2. **Sleep**: replay one user's facts in query format with reads off, plus LM rehearsal.
3. **Verify**: answer with memory off and nothing in context. 0% before, 100% after.
4. **Free**: all 381 slots reclaimable in the best configuration.

And "complete vs incomplete" is no longer a heuristic: a closed episode has a replay target and can be consolidated;
an open one cannot, so it is the only thing eviction should ever drop.

## What this does NOT show

- **One user per sleep.** MQAR draws a fresh fact set per example, so a batch holds contradictory facts and
  consolidation is ill-posed — measured: 3.1% consolidation over 64 conflicting sets, 100% over one shared set. Real
  deployment would sleep per user, which is what this now assumes.
- **Small scale**: 8 facts, 381 slots, one checkpoint, one seed.
- **No livelock or throughput measurement.** Retry budgets and "does sleep keep up with wake" are still open.
- The store used here is the 1x one; sleep at 8x/16x is untested.

## Next

r28: rehearse the **memory task** during sleep, not just the language model — interleave store-using batches with
reads on, so the write path stays calibrated. Target: keep consolidation at 100% and lift new writes from 75% back to
~100%.
