# r12 — does the observation store survive a bigger store?

## Ideate (from r6b, r10, r11)

Everything that works so far used a 512-slot store with two distractor segments (384 slots filled). The objective
the user set earlier is a store that holds a day, and r6b was the only attempt at scale: M=2048 with 14 distractor
segments at batch 8. It kept matching at 100% over 1,920 slots but recall was confounded: the *prediction* store it
used needed in-segment induction, and induction stalled at batch 8.

The observation store (value from the observed token) does not need induction. So r6b's confound is gone and store
size is now measurable.

Memory scales with segments x slots (every segment's `[B, S, M]` read tensors live until backward), so batch must
shrink as the store grows. Two steps, each isolating a 2x and a 4x store:

## Hypotheses

- **H12a — 2x store holds.** M=1024, 8 segments (6 distractor, 896 slots filled), batch 16: single-occurrence recall
  >= 95%.
- **H12b — 4x store holds.** M=2048, 16 segments (14 distractor, 1,920 slots filled), batch 8: >= 95%.
- **H12c — matching stays at 100%** in both, as r6b already showed for 1,920 slots.

## Falsifiers

- Recall drops with matching at 100% -> a content or gate problem that grows with distractor volume.
- Matching drops -> exact token lookup is interfered with at scale (it should not be: keys are token embeddings and
  distractor text uses a disjoint vocabulary from key symbols).
- Both fail to lift off at small batch -> budget/batch, not store size; would need gradient accumulation to
  separate.

## Runs (single occurrence, 8 facts, 6000 steps, seed 0; block-0 observation store — fastest to lift off)

`--key-layer 0 --read-layers 0 --key-pos-free --freeze-query --query-init identity --write-layer 0 --value-shift 1`

| order | name | M | n_seg | batch | slots filled at query |
|---|---|---|---|---|---|
| 1 | r12_M1024 | 1024 | 8 | 16 | 896 |
| 2 | r12_M2048 | 2048 | 16 | 8 | 1,920 |

If both hold, r13 repeats the larger one on the objective placement (last-block write + skip), which lifts off later
and needs more steps.
