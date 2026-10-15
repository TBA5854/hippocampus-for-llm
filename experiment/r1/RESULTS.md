# r1 — results: writing at the last block breaks, and the timing says why

## Numbers (2-fact MQAR, st_bg4 config, one variable changed)

| run | write block | best | final | verdict |
|---|---|---|---|---|
| r1_w4 | 4 | **100.0%** | **100.0%** | st_bg4 reproduced exactly |
| r1_w5 | 5 (last, objective) | 70.5% | **44.4%** | **hypothesis falsified** (< 90%) |
| r1_w5_shuf | 5, scrambled store | 4.2% | 4.2% | chance |
| r1_w5_off | 5, HC off | 4.4% | 4.2% | chance |

Chance is 3.1%. Both controls are at chance at matched settings, so the 44.4% is real memory use —
just much worse and unstable.

## The curves

```
recall   w4  4  2  3 44 81 94 88 99 100 99 100 100 100 100 100 100
         w5  4 43 55 70 68 59 39 14  11 11  27  35  37  42  44  44
seg-0 CE w4 .74 .18 .14 .20 .14 .10 .11 .09 .08 .08 .08 ...
         w5 .73 .41 .23 .16 .14 .12 .11 .10 .10 .09 .10 ...
(one column per 250 steps)
```

Two things stand out:

1. **The last-block write learns far faster at first**: 43% at step 500, when block 4 is still at 2%.
2. **Then it collapses — 70.5% at step 1000 to 11% at step 2250 — exactly while segment-0 loss falls
   from 0.16 to 0.10.** It partly recovers (44%) but never returns.

## Diagnosis

Segment 0 is the facts tiled: `k1 v1 k2 v2 k1 v1 ...`. After the first tile, the model can predict the
next token inside that segment by induction, and the segment-0 loss curve is it learning to.

The last block is what the tied head reads. **Once in-segment prediction is learned, the last block's
state at a position encodes what comes *next*, not what is *here*.** That flips the meaning of both
stored halves under `value_shift=1`:

| slot half | position | block 4 (mid-stack) holds | block 5 (last) holds |
|---|---|---|---|
| key | `k_j` | "this is `k_j`" (+ some prediction) | "next is `v_j`" — varies per example, so a query that only knows `k_j` cannot match it by a fixed projection |
| value | `v_j` (shift 1) | "this is `v_j`" — **the answer** | "next is `k_{j+1}`" — **the wrong token** |

That explains all of it. Early in training the last block still carries token identity, and it is the
most direct thing to store, so it wins fast. As segment-0 prediction sharpens, identity is overwritten by
prediction, the value slot starts carrying the *following* key, and recall collapses. Block 4 keeps
enough identity to survive.

This is a hypothesis, not yet a measurement: r1 ran before the slot diagnostic existed. It makes two
sharp predictions that r2 can check.

- On a rerun of `w5` with the diagnostic, the collapse should show up as **content-miss** (right slot,
  wrong answer), since matching was not what changed.
- The fix follows directly: **at the last block, `value_shift=0`.** The key position's own last-block
  state already says "next is `v_j`", which is exactly the answer.

## What it means for the objective

"Write at the last block before the head" is not wrong, but it is only right if the store is aligned to
what that block represents. The last block is a *prediction*, so a store written there is a store of
predictions: the value lives at the key's own position, not the next one. The mid-stack result
(block 4) is not the architecture the objective asks for; it is a symptom that the shift was set for a
representation of identity.

It also says something useful about keys. A query from layer 1 knows token identity. The last block has
thrown identity away. That is the representational mismatch this design has carried since v0.3, now
measured as a failure rather than assumed as a cost. The natural resolution is attention's own: **take
the key from the same depth as the query, and only the value from the last block.**

## Opens

r2 tests the diagnosis directly, before the fact-count cliff: shift 0 at the last block, and an
early-key / last-value split. The fact-count cliff moves to r3, on whichever placement wins.
