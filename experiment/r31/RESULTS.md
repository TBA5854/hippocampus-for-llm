# r31 — results: episode-aware eviction turns a chance-level store into a working one

## Numbers (128 symbols, 8 facts, cold, 8000 steps, seed 0; chance 0.8%)

| store | write rate | slots/episode | eviction | **recall** | val ppl |
|---|---|---|---|---|---|
| 192 | 25% | ~32 | ring | 100.0% | 2.18 |
| 192 | 25% | ~32 | episode | 88.4% | 4.28 |
| **64** | 25% | ~32 | ring | **0.5%** | 897 |
| **64** | 25% | ~32 | **episode** | **90.6%** | 3.76 |
| 192 | 100% | 127 | ring | 0.5% | 898 |
| 192 | 100% | 127 | episode | 0.5% | 898 |
| **288** | 100% | 127 | ring | **0.5%** | 897 |
| **288** | 100% | 127 | **episode** | **82.5%** | 7.38 |

The two bold pairs are the round's result: same store, same data, same everything but the eviction policy.

**Seed 1 of the headline pair** (288 slots, full rate): ring **1.1%**, episode **100.0%** (val ppl 2.16). So the effect
reproduces, and on this seed the episode policy recovers the store completely.

| seed | ring | episode |
|---|---|---|
| 0 | 0.5% | 82.5% |
| 1 | 1.1% | **100.0%** |

## Verdicts

- **H31a (the ring buffer loses the facts) — confirmed.** Every overflowing ring configuration is at chance (0.5%),
  because the oldest slots are always segment 0 and segment 0 is where the facts are.
- **H31b (episode-aware eviction keeps them) — confirmed, twice.** 0.5% -> **90.6%** at a 64-slot store with compressed
  episodes, and 0.5% -> **82.5%** at a 288-slot store with full-rate episodes. The LM improves with it
  (ppl 897 -> 3.76 and 897 -> 7.38), because the model is no longer training against a store that contradicts itself.
- **H31c (compression and eviction compose) — confirmed.** Compressed episodes are ~32 slots against 127, so four of
  them fit in 128 slots where one full-rate episode barely fits in 192. The best result in the table is the smallest
  store in it.

## The constraint the round found: eviction needs headroom

The M=192 full-rate pair is a **tie at chance** (0.5% both). That is not a policy failure, it is arithmetic: an episode
is 127 slots, so after the complete fact episode there are 65 free, and the very next episode must evict 62 slots when
the only thing in the store is the episode we are trying to protect.

> **A protective eviction policy needs room for the complete episodes plus at least one replaceable one.** Below that
> threshold no policy helps, and the store is simply too small for the workload.

That is worth stating because it is the sizing rule for "a day's worth": capacity must be budgeted per *episode*, not
per slot, and with one episode of slack.

## Diagnosis

The ring buffer's failure is structural rather than statistical. Facts are written first, so they are always the oldest
thing in the store, so an overflowing ring evicts exactly the content the task needs — no amount of training fixes it,
and the diagnostics show a healthy read finding nothing worth reading.

Episode ownership changes what "oldest" means. The allocator spends free slots, then incomplete episodes oldest-first,
and touches a complete episode only when nothing else remains. Completeness is not an arbitrary label: r26/r27 made it
mechanical — a complete episode has a replay target and can be consolidated by sleep, an open one cannot. So the same
distinction governs both eviction and consolidation, which is what the original design asked for:

> "eviction based only replacing stale incomplete stuff but not complete episode which is stale"

## Two bugs this round found, both predicted in the plan

1. **`valid_mask` assumed a prefix.** `idx < n_stored` is true of a ring and false once eviction leaves holes; the plan
   flagged it ("valid_mask must become ep_id >= 0 ... eviction makes holes"). Validity now follows ownership, with a
   test.

   **Correction:** I first attributed the no-overflow gap (88.4% episode vs 100.0% ring at M=192) to this bug. The
   rerun on the fixed build returns **exactly 88.4%**, so that was wrong. With no overflow the two policies write
   identical content — every slot is free, so allocation is sequential either way — and they are equivalent by
   construction. The gap is trajectory variance of the kind r32 measured on this task (8 points of spread across seeds
   for the compressed policy, 72 for full-write), not an effect of the policy. The fix is still correct; it simply had
   nothing to do with this number.
2. **The slot diagnostic crashed the small-store runs.** It maps slot p to position p — false under write selection
   (r30), under episode allocation (r31), and when the store is smaller than one segment. It is now skipped when its
   assumptions do not hold, with a test for each case. It had been producing meaningless numbers since r30; at M=64 it
   started indexing past the store.

## What this does NOT show

- Two seeds for the 288-slot pair; one seed for the 64-slot pair.
- One completeness rule, supplied by the task (segment 0 is complete, distractors are not). A learned write gate
  deciding completeness is the deferred piece.
- No interaction with sleep yet: episodes should also carry their written slot indices, which is r32's fix for the
  broken re-encode path. That is the next step, and it is why the episode table exists in the first place.
