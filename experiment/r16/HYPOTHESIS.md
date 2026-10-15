# r16 — the objective placement at the 4x store, and its control

## Ideate (from r11, r15)

- r11: the objective placement — key + read at layer 0, **value written at the last block with a block-0 skip** —
  reaches 100% on facts seen once at 512 slots (seeds 0 and 1, 32 facts), persisted across processes at 100%.
- r15: the block-0 observation store reaches 100% at the 4x store (1,920 slots) on TinyStories, one-shot read.

The two have not been combined. The skip path lifts off ~3x later than a block-0 value at 512 slots (~2250 vs ~750
steps), and the 4x store delays the block-0 value to ~3750, so the skip path at 4x may need ~10000 steps.

The 4x store also has no scrambled-store control yet.

## Hypotheses

- **H16a — the objective placement holds at 4x.** Last-block write + block-0 skip, 4x store, TinyStories, one-shot
  read: >= 95% within 12000 steps.
- **H16b — the 4x store's control is at chance.** Block-0 observation store, 4x, scrambled store.

## Falsifiers

- H16a stays at chance with matching 100% -> the last-block half of the value swamps the block-0 half once the store is
  large; next: normalise the two halves before summing, or weight the skip.
- H16b above chance -> a leak through the store's structure at scale.

## Runs (single occurrence, 8 facts, TinyStories, one-shot read beta 32, key @0 position-free, frozen identity query,
## shift 1, seed 0, M=2048, 16 segments, 16 x 2 accumulation)

| order | name | value | steps |
|---|---|---|---|
| 1 | r16_skip_M2048 | last block + block-0 skip | 12000 |
| 2 | r16_obs_M2048_shuf | block 0, scrambled store | 6000 |
