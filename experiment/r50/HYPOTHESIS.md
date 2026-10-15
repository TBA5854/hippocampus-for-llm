# r50 — replicate the 100% ladder (seed 2) and push to 64 facts

r48/r49: BPE curriculum reaches 100% at 8/16/32 facts, one seed. Two checks.

- **H50a — the 100% ladder replicates on seed 2.** 4 cold -> 8 warm -> 16 warm on a fresh seed. Falsified if a rung
  lands well below 100% (the char curriculum swung across seeds at low budget; these are better trained, so predicted
  stable).
- **H50b — 64 facts warm from the 32-fact model.** The store is 512 slots; at 64 facts (128 tokens/pair set, tiled)
  it is fuller but likely still not overflowing. Does the ladder hold, or does the store finally bind? First point
  where store capacity (r39 said irrelevant at 8) might re-enter.

box1 = seed2 ladder, box2 = 64 facts. Both need the r49 ckpt on box2 (present) and the bpe json (both boxes).
