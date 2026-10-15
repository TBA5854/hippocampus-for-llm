# r7 — results: on single-occurrence facts, position hijacks the layer-0 key

## What ran

Only the first of four runs completed; the round was stopped by design after its diagnosis showed the
other three would fail for the same reason (they share the key).

| run | value from | shift | best | final | correct-slot reads |
|---|---|---|---|---|---|
| r7_w0s1 | block 0 output | 1 | 5.5% | **3.0%** (chance) | 99.7% at step 750 -> **0.0% from step 1250** |
| r7_w5s1, r7_w5s0, r7_w3s1 | — | — | — | — | stopped before running (see below) |

Correction to the hypothesis doc: `--write-layer 0` takes the value **after** block 0 (attention has run),
not the raw token embedding. So `w0s1` was a "block-0 output at `v_j`" value, not a pure embedding copy.
This does not affect the diagnosis, which is on the key side.

## Verdicts

- **H7b (the ceiling works) — falsified, by a matching failure the hypothesis did not anticipate.** Per its
  own falsifier ("fix the read path before interpreting the rest"), the other runs were stopped.
- **H7a, H7c, H7d — not tested here.** Carried into r8 unchanged, with the fix.

## Diagnosis — position outvotes identity

Correct-slot reads were 85.8% at step 250, 100% at 500, 99.7% at 750, 99.4% at 1000, then **0.0% at step 1250
and every eval after**, with the gate falling 0.092 -> 0.029.

Offline on the final checkpoint (`r7/offline/tax_r7_w0s1.json`, n=320):

| read lands on | share |
|---|---|
| a fact's key (correct or other) | **0%** |
| distractor text in segments 1-2 | 57.8% |
| filler text in segment 0 | 42.2% |

And where those slots sit: the query is at position **126**; the slots read are at positions
**101-126** (sample: 126, 126, 119, 114, 125, 115, 121, 121, 115, 126, 119, 112, 123, 125, 101, 120). The facts
are at positions **0-15**.

A layer-0 state is token embedding + position embedding. With single-occurrence facts the fact keys all
live at the start of segment 0, while the query and hundreds of distractor slots share positions near the
end. As the position embeddings train, positional similarity outgrows token similarity and the read
switches wholesale to slots at the query's own position. Tiled facts never showed this because the key
tokens repeated at every position, including the ones near the query.

## Fix and next round

`--key-pos-free` (commit c15f47c): key and query both from the token embedding alone. Tested: stored keys
equal the token embeddings, and a token matches its slot from a different position. Smoke run: correct-slot
reads 100% at step 20, vs 56% with position included.

r8 reruns r7's four value sources with position-free keys.
