# r31d — the OTHER compression axis: slot width (d_mem)

## Ideate

r30 compressed the *number* of slots: writing the top 25% of positions by surprisal matches a full-write baseline
(100.0% vs 99.7%) and a curriculum reaches 8x (88.3%), with 16x looking like the real floor (~51%, gate wide open,
misses are matching not content).

`d_mem` is the axis the original plan meant by "compression" and the one ask with code already wired and never run.
`--no-write-identity --d-mem N` swaps raw hidden states for the frozen JL projections on the write side; the read side
(`ReadHead.out`) is trainable and has to learn the inverse from the LM loss.

The two multiply: 4x fewer slots at 4x narrower slots is 16x on store bytes.

## Hypotheses

- **H31d-a — d_mem 64 costs little** (>= 95%) at the full write rate. 256 -> 64 is 4x on bytes. Falsified if recall
  falls with width, which would say raw states carry information the projection destroys — the JL argument says it
  should not at these dimensions.
- **H31d-b — d_mem 32 is where it breaks.** 8x on bytes, and the point where a 128-symbol vocabulary plausibly stops
  fitting in the slot.
- **H31d-c — width and rate compose.** d_mem 64 at the 25% write rate holds >= 95%, i.e. 16x total on store bytes with
  no loss. This is the round's real question; the first two are its controls.

## What to watch

The failure signature separates the two stories, as in r30:
- recall falls with **content-miss** high -> the slot is too narrow to hold the value (a true width limit)
- recall falls with the **gate shut** and content-miss low -> the bootstrapping deadlock again, in which case a
  curriculum (wide -> narrow) is the fix, exactly as r30's rate curriculum was.

## A constraint found on launch, and its fix

The first attempt died immediately: `query_init='identity'` requires `d_mem == d_model`, because an identity query
cannot map between different widths. That is not a bug — it is r9's requirement showing up, that query and key must be
comparable. With `write_identity=False` the key is `to_key(h)` for a frozen random projection, so the narrow-store
analogue of the identity query is to give the query **that same frozen projection**: `--query-init key_proj`, added
this round with a test. A learned query is not an option — r8 measured it being steered away from facts.

## Runs (box2, 1x store, 128 symbols, 8 facts, cold, 8000 steps; the r23 configuration)

| name | d_mem | write rate | store bytes vs baseline |
|---|---|---|---|
| r31d_w64 | 64 | 100% | 1/4 |
| r31d_w32 | 32 | 100% | 1/8 |
| r31d_w64_r25 | 64 | 25% (surprisal) | **1/16** |

Baseline for all three: r23's 99.7% at d_mem 256, full rate.
