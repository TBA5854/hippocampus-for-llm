# r19 — warm-start the 4x last-block write from the 2x model

## Ideate (from r16-r18)

The objective placement (read @ layer 0, write @ last block with a normalised block-0 skip) reaches 99.4% at a 2x store
(r18) but did not lift off from scratch at 4x in 12000 steps (r17). Every failed 4x run finds the right slot (100%) but
never starts using it; read mass decays while it waits. The difficulty looks like *discovering* how to use the read, not
representing the 4x store.

`--init PATH` (new) loads a checkpoint before training; store tensors are non-persistent buffers, so a 2x-trained model
loads into a 4x model unchanged. CPU smoke test: `r18_skipnorm_M1024` loaded into a 4x model gives **52.5% recall at step
10** and 62.5% at step 20 — the read transfers across store sizes almost for free.

## Hypotheses

- **H19a — warm start reaches >= 95% at 4x** within 4000 steps (curriculum over store size).
- **H19b — the warm-started model still reads the store's content.** The same initialisation with a scrambled store is
  at chance. Without this, a high number could come from the warm-started weights exploiting something other than the
  facts in the store.

## Falsifiers

- Warm start degrades toward chance during 4x training -> the 2x solution is unstable at 4x (the drift that blocked the
  cold runs wins); try a lower learning rate for the warm start.
- Scrambled control above chance -> the transferred model answers without the store's content.

## Runs (TinyStories, single occurrence, 8 facts, one-shot read beta 32, key @0 position-free, frozen identity query,
## write @ last block + normalised block-0 skip, shift 1, M=2048, 16 segments, 16 x 2, 4000 steps, seed 0,
## --init experiment/r18/ckpt/r18_skipnorm_M1024.pt)

| order | name | store |
|---|---|---|
| 1 | r19_warm_skipnorm_M2048 | real |
| 2 | r19_warm_skipnorm_M2048_shuf | scrambled |
