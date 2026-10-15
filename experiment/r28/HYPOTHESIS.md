# r28 — keep the write path calibrated through sleep

## Ideate (from r27)

The eviction loop works: decode the store, replay with reads off, verify with memory off, free what verifies.
Consolidation 0% -> 100%, all 381 slots reclaimable, LM loss *improved* (0.961 -> 0.833) by rehearsal. Two costs were
left on the table:

1. **New writes degrade to 75%.** Refilling the store with post-sleep weights reads back worse than before. Sleep
   rehearses the language model but nothing rehearses the *memory task*, so the write path (last block + block-0 skip)
   drifts away from the read path with no pressure to stay aligned.
2. **Drift past ~400 destroys the store entirely** (0% readable). Not addressed here; it is a budget limit, and the
   working band is drift <= ~120.

r27 also showed freezing the key space is the wrong tool — it holds old keys valid but decalibrates write from read
(25% new writes). So the fix should keep everything moving and add the missing pressure instead: interleave real
memory-task batches during sleep, reads ON, on *fresh* facts (`--rehearse-mem`, added this round).

## Hypotheses

- **H28a — memory rehearsal restores new writes.** With `--rehearse-mem 1.0`, refilling the store after sleep reads
  back >= 95% (r27's best: 75%), while consolidation stays at 100%. Falsified if new writes stay ~75%: then the
  degradation is not a calibration drift but something the write path cannot recover by rehearsal.
- **H28b — it does not cost consolidation.** Memory rehearsal keeps reads alive for part of each step, and the whole
  point of SLEEP's reads-off rule is that a live store lets the model answer by lookup instead of absorbing. If
  consolidation falls below ~75% the two objectives are in conflict and the rehearsal must be interleaved *between*
  sleeps rather than inside one.
- **H28c — the loop survives at the 8x store** (3,937 slots, box1). Everything so far is the 1x store. Falsified if
  consolidation or decoding degrades with store size.

## Runs

| box | name | change |
|---|---|---|
| box2 | r28_mem0 | rehearsal only, 1500 steps, lr 3e-4 — r27's best, as the reference line |
| box2 | r28_mem1 | + `--rehearse-mem 1.0` |
| box2 | r28_mem3 | + `--rehearse-mem 3.0` |
| box2 | r28_mem1_s4000 | `--rehearse-mem 1.0`, 4000 steps (does more budget close it?) |
| box1 | r28_8x_mem0 | r20's 8x checkpoint, rehearsal only — does the loop hold at 3,937 slots? |
| box1 | r28_8x_mem1 | 8x + `--rehearse-mem 1.0` |

All with `--shared-facts` (r27: consolidation is ill-posed across contradictory fact sets) and n=64 queries.
