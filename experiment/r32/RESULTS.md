# r32 — results: compression and sleep interact, and the interaction is in the REPAIR path

## Part 1 — sleep on a compressed store (box1, r30's 25% checkpoint, 96 slots/segment, n=64)

| measurement | full store (r28) | **compressed (25% surprisal)** |
|---|---|---|
| consolidation: held-out fact from weights, memory off | 0% -> 100% | 0.125 -> **100%** |
| LM loss | improves | 0.948 -> **0.829** |
| pre-sleep store still readable | 100% | 62.5% |
| store REWRITTEN after sleep | 100% | **10.9%** |
| store RE-ENCODED from its own transcript | 100% | **6.3%** -> **75.0%** (r33/r33b, below) |
| weight drift | ~120 | 120 |

`decode_exact` reads 0.26 and should be ignored: it is the invalid-alignment artefact named in r30 (it assumes slot i
holds position i, which selection breaks). Consolidation reaching 100% is the proof that replay itself worked.

## Diagnosis — a content-dependent write policy is not stable across sleep

Consolidation is unaffected by compression: the store drains into the weights exactly as it did at full rate, and the
LM improves. That was the open question from r30 (coverage said replay *should* survive) and it is answered yes.

What breaks is **re-writing the store afterwards**, which r28 established as the repair for key drift. The mechanism is
specific and was not visible before:

> The surprisal gate selects positions by *the model's own next-token NLL*. Sleep changes the model. So after sleep the
> gate selects a **different subset**, and the re-encoded store is not the store that was decoded.

A full-write policy has no such dependency — every position is written regardless of what the model believes — which is
why r28 measured 100% on the same operation. Compression buys 4x fewer slots at the cost of making the write policy a
*function of the weights*, and the sleep loop's repair step assumed the policy was fixed.

This is a design finding rather than a bug: the eviction loop as specified (decode -> sleep -> verify -> free ->
re-encode the rest) is sound for a fixed write policy and needs one change for a content-based one — **re-encode with
the pre-sleep scores**, i.e. carry the selected indices with the episode instead of recomputing them. The store already
knows which positions it wrote; nothing needs to be re-derived. Untested, and the obvious r33.

## r33 / r33b — the fix, and a correction to the diagnosis above

Two runs off the *same* checkpoint as Part 1. Everything but the re-encode is bit-identical to r32 (same decode,
same consolidation, same LM loss, weight drift agreeing to five decimals), so this is as clean a controlled
comparison as this project has produced.

| re-encode policy | slots written | **re-encoded store** | facts recovered |
|---|---|---|---|
| r32: re-score the transcript with the 25% gate | ~24 | **6.3%** | 0.5 / 8 |
| r33: replay the episode's recorded positions | ~24 | **75.0%** | 6 / 8 |
| r33b: rewrite the transcript 1:1 (`reencode_policy: all`) | 95 | **75.0%** | 6 / 8 |

### The diagnosis above was incomplete, and the arithmetic says why

`decode_store` returns **one token per slot, in slot order**, so the transcript is 96 tokens *in slot coordinates*.
Check it against the run config: `seg_len 128`, `write_frac 0.25`, `value_shift 1`, and wake writes `n_seg - 1 = 3`
segments, so `round(0.25 x 127) x 3 = 96` — exactly the reported `n_slots`.

**The transcript is already the compressed selection.** Re-running the 25% gate over it therefore compresses an
already-compressed store a *second* time, 96 slots down to 24. r32 read that as policy instability; it is more
specifically a **double compression**, and the framing matters because it changes the fix.

It also makes r32's number mechanical rather than merely bad: after consolidation the facts are the tokens the model
predicts **best**, so a surprisal gate ranks them *lowest* and discards precisely the content worth keeping. That is
r6's inversion — don't store what the model can predict — biting from the other side.

The r33 fix (carry the written indices) was itself built on a coordinate error: `slot_pos` records positions in the
original 128-token wake segment, while the transcript is 96 tokens in slot space. Forcing one onto the other indexed
past the end and crashed as a CUDA device-side assert; the bounds check added afterwards silently dropped ~3/4 of the
positions, and the run scored 75% because the survivors happen to land on the fact region. Right number, wrong
mechanism. **The correct re-encode runs no selection policy at all** — the transcript is the selection, so it is
rewritten one slot for one. That is what r28's full-write policy did implicitly when it measured 100% here.

### What the tie means — the residual is NOT the write policy

r33 wrote ~24 slots and r33b wrote 95, and both land on **exactly 75.0%**. With 8 shared facts over 64 examples the
metric has only 8 levels (8/64 = 0.125 each), so 75% is "6 of 8 facts" and the tie is a coarse measurement landing in
one bucket rather than a coincidence. But two write policies differing 4x in slot count giving the same answer says
the remaining gap to r28's 100% **is not in the re-encode policy**. It is upstream, in how faithfully a compressed
store decodes at all: `decode_exact` reads 0.26, and while that figure is partly the invalid-alignment artefact r30
named, re-encoding cannot restore what did not decode. Measuring decode fidelity on a compressed store *with a valid
alignment* is the next question, and it was not on the list before this round.

## Part 2 — seed 2 of the compression claim (box2)

| write policy | seed 0 | seed 1 | seed 2 | mean | spread |
|---|---|---|---|---|---|
| **25% surprisal** | **100.0%** | **100.0%** | **91.9%** | **97.3%** | 8 pts |
| 100% (write everything) | 99.7% | 27.8% | 85.8% | 71.1% | **72 pts** |

**The claim is about variance, not collapse.** Seed 1's 27.8% was not a broken run — it is the tail of a wide
distribution. Across three seeds the compressed policy is both better on average (97.3% vs 71.1%) and far more stable
(8 points of spread vs 72). Every run had `slot@1` at 100%, so the variation is entirely in how quickly the read
becomes usable, not in whether retrieval works.

That is the r5 correlated-key story once more: a store of near-duplicate keys can be learned from, but *whether* it is
learned in a given budget depends on the draw. Removing the duplicates removes the gamble.

## What this does NOT show

- One sleep cycle on one compressed checkpoint. The multi-cycle steady state (r29) has not been rerun compressed.
- 75% is not 100%, and the 25% shortfall is unexplained. The tie between two 4x-different write policies locates it
  upstream of the re-encode, but "upstream" is a direction, not a measurement.
- One seed per policy. The three re-encode numbers come from one checkpoint.
- `decode_exact` still has no valid alignment under a selective write policy, so the decode-fidelity hypothesis for
  the residual is currently reasoning, not data.
- `post_sleep_store_recall` at 62.5% is itself lower than r28's 100%; with a content-based policy even the *existing*
  store is partly unreadable after sleep, for the same reason — the read is matched against keys the current model
  would no longer choose to write.
