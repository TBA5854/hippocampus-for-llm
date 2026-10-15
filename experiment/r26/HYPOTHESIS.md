# r26 — can a stored value be decoded back to a token id?

## Ideate (from r25, and the sleep design)

r25 closed the training-time story: store size, fact count and key vocabulary are one bootstrapping failure, and a
curriculum escapes it. What is still missing is the *deployment* story. At wake the weights are frozen, so every new
fact lives in the store — and the store is finite and evicts. Anything that must outlive the store has to move into the
weights. That is consolidation, and `mech/wake_sleep.py` already implements it: reads disabled, weights unfrozen,
replay, then probe with memory off and an empty prompt (a correct answer there can only have come from weights).

Its stated limitation is the blocker: replay is *reconstructed from a known persona*, not decoded from the store,
"because the store holds pooled d_mem summaries and no token ids". True of the v0.1-v0.3 store. The current one writes
per token, and the value carries the **observed** token (`--value-shift 1`) with a skip from block 0 — the layer where
token identity still lives (r9: block-0 values hold facts seen once at 100%, last-block values do not).

If stored values decode back to token ids, sleep can replay from the store itself, and eviction becomes
**consolidate -> verify with reads off -> free**, with episode completeness falling out rather than being stipulated:
a closed episode has a replay target, an open one does not.

## Hypotheses

- **H26a — stored values decode to their observed token with no new parameters**, via nearest neighbour against the
  token embedding matrix or the model's own (tied) LM head: >= 90% exact on the fact symbols. Falsified if both sit
  near the unigram rate — then replay needs a trained decoder.
- **H26b — a linear probe decodes them** (upper bound on any learned decoder): >= 95% on held-out slots. Falsified
  too -> the value does not carry token identity at all and sleep cannot free slots on its own; the store would need a
  parallel token buffer, as v0.1 had.
- **H26c — decoding is better on fact symbols than on distractor text.** The write gate is supposed to favour facts;
  if text decodes as well as symbols, that is worth knowing before building replay on top of it.

## Why the split in H26c matters

A decoder that always says " " (the most common character) would look strong on a store that is mostly English text.
`mech/decode.py` therefore reports accuracy separately on fact symbols and on text, plus the symbol fraction of the
store, so a degenerate decoder cannot pass.

## Runs (box1, on r20's 8x checkpoint — the largest store with a matching results JSON)

| name | what |
|---|---|
| r26_decode_8x | `mech.decode --run r20/results/r20_warm_skipnorm_M4096.json --ckpt r20/ckpt/r20_warm_skipnorm_M4096.pt` |
| r26_decode_1x_f48 | same probe on r25's 48-fact 1x checkpoint — many more symbols per store, so it tests the harder case |

Both are single forward passes plus a 300-step linear fit: minutes, not hours.

## If it works

Port `wake_sleep.py` to the current architecture (per-token store, read at layer 0, write at the last block with the
block-0 skip) and drive SLEEP's replay from decoded store contents instead of a known persona. Then measure the
eviction loop end to end: fill, sleep, probe with reads off, free only what the probe confirms, and report how many
slots each sleep actually reclaims.
