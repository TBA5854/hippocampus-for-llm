# r28 — results: memory rehearsal repairs the write path, and ages the stored keys

## Numbers (`mech/sleep_evict.py`, `--shared-facts`, 1500 steps at lr 3e-4 unless noted)

| run | store | n | rehearse-mem | **consolidation** | old store | new writes | LM | drift |
|---|---|---|---|---|---|---|---|---|
| r28_mem0 (r27's best) | 1x | 64 | — | **100%** | 88% | 75% | 0.833 | 121 |
| r28_mem1 | 1x | 64 | 1.0 | **100%** | 38% | 86% | 0.807 | 126 |
| **r28_mem3** | 1x | 64 | 3.0 | **100%** | 38% | **100%** | **0.792** | 128 |
| r28_mem1_s4000 | 1x | 64 | 1.0 (4000 steps) | **100%** | 25% | 75% | 0.601 | 210 |
| r28_8x_n16_mem0 | 8x | 16 | — | **100%** | 100% | 62% | 0.838 | 105 |
| r28_8x_n16_mem1 | 8x | 16 | 1.0 | **100%** | 75% | **100%** | 0.734 | 104 |
| r28_8x_mem0 | 8x | 8 | — | 100% | 100% | 62% | 0.901 | 103 |
| r28_8x_mem1 | 8x | 8 | 1.0 | 100% | 100% | 88% | 0.774 | 101 |

LM loss before sleep: 0.961 (1x), 0.823 (8x). Consolidation was 0% before sleep in every run. Decoding was exact
(100%) in every run.

## Verdicts

- **H28a (memory rehearsal restores new writes) — confirmed.** 75% -> **100%** at weight 3 (1x) and at weight 1 (8x,
  n=16). The degradation was a calibration drift, and rehearsing the memory task is what re-calibrates it.
- **H28b (it does not cost consolidation) — confirmed.** 100% in every single run, at every rehearsal weight. Keeping
  reads alive for part of each step does not let the model "answer by lookup instead of absorbing", which was the risk.
- **H28c (the loop holds at the 8x store) — confirmed.** 3,937 slots: consolidation 0% -> 100%, decoding exact, and
  the same pattern as 1x. Ten times the scale, same behaviour.
- **New: memory rehearsal ages the stored keys.** Old-store readability falls 88% -> 38% (1x) and 100% -> 75% (8x) as
  rehearsal rises. More budget makes it worse (4000 steps: 25%, drift 210).

## Diagnosis — the key space cannot be in two places at once

Put r27 and r28 together and the tension is exact:

| what sleep does to the key space | old store | new writes |
|---|---|---|
| freeze it (r27) | **100%** | 25% |
| leave it (r28 mem0) | 88% | 75% |
| actively rehearse writes into it (r28 mem3) | 38% | **100%** |

Stored keys are token embeddings. Reading them back requires the embedding to be where it was at write time; writing
*new* memories correctly requires the embedding to be where the rest of the network now is. Sleep moves the network.
So every setting trades the store it has against the store it will write next.

**The resolution is already in these numbers, and it is specific to this architecture.** `new writes` is measured by
refilling the store under the post-sleep weights — and r26 showed the store decodes to exact token ids. So a surviving
episode does not need its original text to be rescued: it can be **re-encoded from its own transcript**. At mem3 that
path reads back at 100%. Key drift is repairable, not fatal.

That closes the loop as a design:

1. **Decode** the store to token ids (exact, no decoder).
2. **Sleep**: replay in query format, reads off, with LM rehearsal *and* memory rehearsal.
3. **Verify** with memory off and an empty prompt; **free** what verifies (here: everything).
4. **Re-encode** whatever did not consolidate, from its transcript, under the new weights.

Step 4 is what makes steps 2-3 safe to run on a live store. `mech/sleep_evict.py` now measures it directly
(`store_reencoded_from_transcript`), added after these runs — the number reported above as "new writes" is the same
operation applied to the whole store.

## What this does NOT show

- **One user per sleep** (r27): consolidation is ill-posed across contradictory fact sets.
- **Single cycle.** Every run here is one wake -> sleep. Repeated cycles — does drift accumulate, does the model
  survive ten sleeps — are untested, and that is the next thing the design needs.
- Small evals at 8x (n=8 and n=16, because store memory forces small batches; batch 32 OOMs at 15.6 GB). The n=8
  "old store 100%" for mem1 did not survive the n=16 rerun, which gave 75% — a caution about reading the 8x column
  too precisely.
- 8 facts, one checkpoint per store size, one seed.

## Next

r29: the **steady-state loop** — repeated wake/sleep cycles with re-encoding in place. Measure drift accumulation
across cycles, slots reclaimed per cycle against slots filled per wake (throughput), and a retry budget for facts that
never consolidate (livelock). That is the last open piece of the eviction design.
