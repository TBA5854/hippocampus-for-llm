# r2 — align the store to what the last block represents

## Ideate (from r1)

Writing at the last block (block 5) reached 70.5% by step 1000, then collapsed to 11% while segment-0
loss fell, ending at 44.4%. Block 4 held 100%. The diagnosis: once the model learns to predict inside
the tiled fact segment, **the last block encodes what comes next, not what is here**. Under
`value_shift=1` that makes the value at `v_j`'s position say "next is `k_{j+1}`" (the wrong token), and
makes the key at `k_j` say "next is `v_j`", which a layer-1 query knowing only `k_j` cannot match.

## Hypotheses

- **H2a — the collapse is content, not matching.** Rerunning r1's `w5` with the slot diagnostic, the
  collapse shows up as **content-miss** (read lands on the right slot, answer wrong).
- **H2b — shift 0 fixes the value.** At the last block the key position's own state already says "next
  is `v_j`". With `value_shift=0`, `w5` should not collapse.
- **H2c — early keys fix the matching.** Taking the key from the query's own depth (layer 1, pre-read)
  and only the value from the last block gives keys that carry token identity, exactly like attention's
  `W_q`/`W_k` at one depth. Prediction: the best of the four, stable, and at least as fast as `w5`'s
  early lift-off.

## Falsifiers

- `w5s1` collapse is **match-miss** -> the diagnosis is wrong about which half broke.
- `w5s0` still collapses -> the value is not the (only) problem.
- `w5s0k1` is no better than `w5s0` -> keys were never the bottleneck at 2 facts. (It may still matter
  more at 8 facts, which is r3's job.)
- **Instrument check:** `w4` at 100% recall must show `slot@1` near 100%. If not, the slot mapping is
  wrong and no diagnostic number this round is usable.

## Runs (2 facts, 4000 steps, st_bg4 config otherwise; parallel)

| name | write | key from | shift | purpose |
|---|---|---|---|---|
| r2_w4 | 4 | write | 1 | reference + instrument check |
| r2_w5s1 | 5 | write | 1 | r1_w5 again, with the diagnostic (H2a) |
| r2_w5s0 | 5 | write | 0 | H2b |
| r2_w5s0k1 | 5 | layer 1 | 0 | H2c |
| r2_w5s0k1_shuf | 5 | layer 1 | 0 | scrambled-store control for the new config |
