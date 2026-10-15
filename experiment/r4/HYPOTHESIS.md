# r4 — lock the 8-fact result, then budget vs capacity at 16

## Ideate (from r3)

r3 put 8-fact recall at 83.9% with the value written at the last block. Before building on it, the
objective's own bar needs the pieces r3 did not have: an HC-off control at this config, more than one
seed, and a persisted store. And the 16-fact failure has two readings that predict different futures.

## Hypotheses

- **H4a — the 8-fact result is not a seed accident.** Seeds 1 and 2 land within ~10 points of 83.9%.
- **H4b — HC-off at 8 facts is at chance** (~3%), completing the control pair with r3's scrambled 2.7%.
- **H4c — 16 facts is budget, not capacity.** 8 facts sat at chance until step ~1750 then lifted off; lift-off
  time grows with fact count (fewer correct slots, weaker signal). With 12k steps, 16 facts lifts off.
- **H4d — the matching misses are not random** (offline, on r3's checkpoints). The ~12% of reads that
  miss land somewhere structured — another fact's key, a value slot, or distractor text — and which one
  names the next fix.
- **H4e — the store persists** (offline). A store written from segment 0, saved to disk, and loaded into
  a fresh process answers the query segment with nothing else in context, at the in-process recall rate.

## Falsifiers

- A seed below ~60% -> 83.9% was lucky; the headline becomes the seed mean with its spread.
- HC-off above chance -> a leak: the query segment is answerable without the store.
- f16 at 12k still at chance -> capacity (or a signal problem more steps cannot fix); next lever is the
  gradient floor or a curriculum.
- Reloaded recall below in-process recall -> something the store needs (e.g. `key_mean`) is not being
  persisted.

## Runs (sequential)

| order | name | purpose | ~time |
|---|---|---|---|
| 1 | r4_f8_off | H4b | 11 min |
| 2 | r4_f8_s1 | H4a | 11 min |
| 3 | r4_f8_s2 | H4a | 11 min |
| 4 | r4_f16_12k | H4c | 34 min |

Offline while those train: miss taxonomy (H4d) and persistence (H4e) on `r3/ckpt/r3_f8.pt`.
