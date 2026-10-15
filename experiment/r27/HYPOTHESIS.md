# r27 — does sleep actually reclaim slots, and what does it cost?

## Ideate (from r26 and a first smoke test)

r26 showed the store is a readable transcript: keys decode to token ids at 100%, and slot i+1's key is slot i's value
token, so replay needs no decoder. `mech/sleep_evict.py` (written this round) runs the loop end to end:
WAKE (frozen weights, store answers) -> DECODE (keys -> tokens) -> SLEEP (reads off, weights move) -> VERIFY (reads
off, nothing in context) -> FREE (only what VERIFY confirms).

A first smoke test on the 48-fact checkpoint, naive settings (400 steps, lr 1e-4, no protection), failed three ways at
once and each failure taught something:

| measurement | result | what it means |
|---|---|---|
| replayed fact, from weights | 100% | rote: it was the one query sleep supervised |
| **held-out fact, from weights** | **0%** | "k v" in replayed text does NOT teach `<query> k -> v` |
| store rewritten after sleep | 25% | sleep damaged the memory mechanism itself |
| LM loss | 0.89 -> 1.72 | catastrophic forgetting on a tiny replay set |

Three fixes follow directly, all now implemented:

1. **Format-matched replay.** The store hands us every pair, so synthesise one supervised query per fact, each with
   fresh distractor text — not plain LM text over the transcript.
2. **`--freeze-keys`.** Matching is nearest neighbour in the token-embedding space, and the stored keys were written by
   the pre-sleep embedding. If the embedding moves, slots not yet consolidated become unreadable: consolidation would
   destroy the memory it is draining. Freezing the embedding and block 0 pins that space.
3. **`--rehearse`.** Interleave general LM batches — the standard guard against forgetting.

With 2 and 3 the LM damage is gone (0.90 -> 0.89) and rewritten-store recall rises 25% -> 75%, but consolidation itself
did not happen in 200 steps. That is the trade-off this round measures.

## Hypotheses

- **H27a — protected sleep consolidates.** With frozen keys + rehearsal and enough budget, held-out facts reach >= 50%
  from weights alone with memory off. Falsified if it stays at 0% at every budget: then the frozen key path is *why*
  nothing transfers, and consolidation and store-readability are in direct conflict.
- **H27b — protected sleep keeps the store readable**, i.e. the pre-sleep store still answers >= 90% after sleep. The
  smoke test could not tell: sleep's own forward passes reset the store, so it was measuring an emptied buffer, not
  key drift. The store is now snapshotted and restored before that measurement.
- **H27c — LM loss is preserved** within ~0.05 with rehearsal on.

## Falsifiers worth naming

- Consolidation rising only as store-readability falls -> the loop cannot be run on a live store; sleep would have to
  drain it in one pass or work on a copy.
- Held-out facts at 0% while replayed facts hit 100% -> the model memorises sequences, not associations, and replay
  needs a different format again.

## Runs (box2, on r23's 8-fact 1x checkpoint — the easiest case)

| name | protection | steps | lr |
|---|---|---|---|
| r27_naive_s400 | none | 400 | 1e-4 |
| r27_prot_s400 | keys frozen + rehearsal | 400 | 1e-4 |
| r27_prot_s1500 | same | 1500 | 1e-4 |
| r27_prot_s1500_lr3e4 | same | 1500 | 3e-4 |
| r27_prot_s4000_lr3e4 | same | 4000 | 3e-4 |
