# r55 — does dedup-at-write fix surprisal's repeat-suppression? (16-fact case)

r54 found surprisal-eviction matches the oracle for 64 *distinct* facts (clean threshold M\*≈512/15win)
but plateaus at ~75% for 16 *tiled* facts at any M, because surprisal scores **novelty**: the fact
window `k1 v1 … k16 v16` is tiled ~4× to fill 128 tokens, and every repeat after the first is now
predictable → low surprisal → evicted like prose. So a fact can lose all its surviving copies.

The fix the mechanism implies: **deduplicate**. Among slots holding the same token, keep only the
highest-surprisal (i.e. first) copy before the top-M keep. Repeats stop crowding the ranking; each fact
is represented once, by its high-surprisal first occurrence.

## Setup — offline, no new training

New policy `surprisal_dedup` added to `mech.persist evict`: per example, null the surprisal of any slot
whose token has a higher-surprisal twin, then keep the top-M. Compared against `oracle` (upper bound)
and plain `surprisal`. Same M grid and bridges as r54.

    uv run python -m mech.persist evict --run ... --ckpt ... \
        --policies oracle surprisal surprisal_dedup --nseg-list 16 32 \
        --mem-list 256 384 512 768 1024 1536 2048 --out r55_<tag>.json

- **box1 (main):** 16-fact `r50_bpe16_s2` — the case r54 falsified. The test.
- **box2 (control):** 64-fact `r50_bpe64` — no tiling repeats, so dedup should ≈ plain surprisal;
  confirms dedup does not *hurt* the distinct case.

## Hypothesis

**Claim:** `surprisal_dedup` lifts the 16-fact plateau toward the oracle and restores a clean M\*, and
leaves the 64-fact curve essentially unchanged.

**Predict:** 16f dedup ≫ 16f surprisal (75% → ~oracle above some M\*); 64f dedup ≈ 64f surprisal.

**Falsified if** dedup does not lift 16f (then the plateau is not repeat suppression but something else
— e.g. the read cannot resolve a fact once its tiled redundancy is gone, or prose first-occurrences
still outrank facts), or if dedup *hurts* 64f (removing a useful redundant copy). Either falsification
is a real finding about what the store needs beyond "keep one copy".

## What it decides
Dedup works → the deployable rule sharpens: **store each fact once and evict by surprisal**; the r54
margin law then applies regardless of how often a fact was restated. Dedup fails → the plateau is
deeper than novelty, and the learned binding-aware gate (r56) is the necessary route.
