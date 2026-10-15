# r26 — results: the store is exactly readable, through its keys

## Numbers (`mech/decode.py`, box2, on two trained checkpoints; 381 slots per example)

| decoder | needs training? | f48 symbols | f48 text | f8 symbols | f8 text | 8x symbols | 8x text |
|---|---|---|---|---|---|---|---|
| **key, nearest neighbour vs the embedding matrix** | no | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |
| **value token read off the NEXT slot's key** | no | **100%** | 99.6% | **100%** | 99.6% | **100%** | 99.3% |
| value, nearest neighbour | no | 0.0% | 4.4% | 0.0% | 4.3% | 0.0% | 6.1% |
| value, the model's own (tied) LM head | no | 0.0% | 4.5% | 0.0% | 4.4% | 0.0% | 6.0% |
| value, linear probe fitted on one batch | yes | 82.8% | 99.0% | 22.5% | 99.5% | 28.3% | 99.8% |

`f48` = r25's 48-fact 1x checkpoint (381 slots, 25% fact symbols), `f8` = r23's 8-fact one (381 slots, 3.9% symbols),
`8x` = r20's 8x store (**3,937 slots**, 0.4% symbols). The key result holds from 381 slots to 3,937: exact at every
size, on symbols and on English text alike.

## Verdicts

- **H26a (values decode with no new parameters) — falsified, decisively.** 0.0% on fact symbols by either
  parameter-free route. The stored value is `scale * (unit(last block) + unit(block 0))`; that mixture is not in the
  embedding space, and its last-block half encodes the *next* token rather than the one written.
- **H26b (a linear probe decodes values) — partially confirmed, and now moot.** 82.8% on symbols in the dense store,
  22.5% in the sparse one — the sparse number is starvation (only ~120 symbol slots to fit 128 classes from), not a
  different representation. Either way this is the wrong instrument.
- **H26c (symbols vs text) — confirmed and important.** The overall probe figure of 95-96% is almost entirely text.
  Splitting was what exposed that the parameter-free value decoders get *nothing* right.
- **New, and the actual result: the keys decode exactly.** A position-free layer-0 key *is* the token embedding, so
  nearest neighbour recovers it at 100% — on symbols and on English text alike, at both checkpoints.

## Diagnosis — replay needs no decoder at all

The pairing is `(key at position i, value at position i+1)`, so **slot i+1's key is slot i's value token**. The key
sequence alone reconstructs the written text. Measured that way, the value token is recovered at 100% on fact symbols
and 99.6% overall — the missing 0.4% is the segment boundary, where a segment's last slot has no successor inside it.

So the store is not an opaque bag of vectors. It is a literal, ordered transcript of what was written, readable with
one matrix multiply against weights the model already has. `wake_sleep.py`'s documented limitation — "replay is
reconstructed from the persona, not decoded from the store... true replay-from-store needs the parallel token buffer
v0.1 had" — is resolved, and without the token buffer: position-free keys made the store self-describing.

A caveat worth keeping: this works *because* keys are token embeddings, which is the same property that makes
retrieval near-exact token lookup (the "retrieval by meaning" limit in OBJECTIVE.md). Contextual keys would buy
meaning-based retrieval and would cost exact decodability. The two are the same design choice seen from opposite
sides.

## What this makes possible

Sleep as the eviction mechanism, end to end and with no new machinery:

1. **Read** the episode out of the store — keys -> token ids, in slot order.
2. **Consolidate**: replay that text with reads disabled and the weights unfrozen (`wake_sleep.py`'s SLEEP phase).
3. **Verify**: probe with reads off and nothing in context. `wake_sleep.py` already has this probe, plus an LM-loss
   check that catches consolidation damaging general ability.
4. **Free** only the slots the probe confirms; leave the rest for the next sleep.

Episode completeness stops being a heuristic: a closed episode has a replay target, an open one does not, so "evict
stale incomplete episodes, never complete ones" becomes "consolidate what can be consolidated, drop what cannot".

Two risks to design against, neither measured yet: **livelock** (facts that never consolidate pin the store — needs a
retry budget) and **throughput** (sleep must reclaim slots at least as fast as wake fills them).

## Next

r27: port `wake_sleep.py` to the current architecture (per-token store, read at layer 0, write at the last block with
the block-0 skip) with replay decoded from keys, then measure the loop: slots reclaimed per sleep, recall after
consolidation with reads off, and LM loss before/after.
