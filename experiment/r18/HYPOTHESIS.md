# r18 — where does the last-block write stop working as the store grows?

## Ideate (from r16, r17)

The objective placement (read at layer 0, value written at the last block with a block-0 skip):

| store | corpus | result |
|---|---|---|
| 1x (384 slots) | tinyshakespeare | **100%** (r11, 8000 steps, lift-off ~2250) |
| 4x (1,920 slots), plain skip | TinyStories | chance at step 5000, gate closing (r16, stopped) |
| 4x, normalised skip | TinyStories | chance through step ~9500 of 12000, gate flat ~0.02, mass 0.46 (r17) |

The block-0 value alone works at every size tried (1x/2x/4x). So the skip path breaks somewhere between 1x and 4x, and
two things changed at once: store size and corpus. Rather than pay 2+ hours per 4x attempt, r18 finds the break on one
corpus with cheaper runs.

## Hypotheses

- **H18a — the skip path still works at 1x on TinyStories.** (Separates the corpus change from the store change.)
- **H18b — it works at 2x.** If so, the failure is between 2x and 4x (budget/drift at the largest size); if not, it
  starts earlier.
- **H18c — normalising helps at 2x** if the plain skip fails there.

## Falsifiers

- 1x fails on TinyStories -> the corpus, not the store, broke the skip path (e.g. the last-block state carries more
  from richer text and swamps the skip); investigate at 1x where runs are cheap.
- 2x plain and norm both fail -> the break is between 1x and 2x.

## Runs (TinyStories, single occurrence, 8 facts, one-shot read beta 32, key @0 position-free, frozen identity query,
## write @ last block + block-0 skip, shift 1, seed 0, 8000 steps)

| order | name | M | n_seg | batch | skip |
|---|---|---|---|---|---|
| 1 | r18_skip_M512 | 512 | 4 | 32 | plain |
| 2 | r18_skip_M1024 | 1024 | 8 | 32 | plain |
| 3 | r18_skipnorm_M1024 | 1024 | 8 | 32 | normalised |
