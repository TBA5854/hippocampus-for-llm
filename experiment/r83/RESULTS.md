# r83 — route A (CoT token scratchpad), fixed masking: RESULTS

**Route A retrieval is real (the r81 "failure" was a masking bug). hop1 works at
~90%; hop2 caps at ~45% = 0.5 × 90% because the intermediate token collides in the
store. Mechanism sound; a store-collision caps the second hop.**

## Numbers (final, 8k, teacher-forced; per-hop = char pos split)
| arm | recall (joint) | hop1 | hop2 |
|-----|----------------|------|------|
| r83_chain (seed 0) | 38.1% | 89% | 45% |
| r83_chain_s1 (seed 1) | 38.9% | 90% | 45% |

Replicates across seeds. hop2 plateaued (48% at step 300 → 45% at 8k — did not
climb).

## Diagnosis: the intermediate-token store collision
hop2 = 45% ≈ **0.5 × 90%**. The intermediate t1 is the VALUE of `t0→t1` and the KEY
of `t1→t2`. With value_shift, t1's value-occurrence (in "t0 t1 pad") makes a spurious
slot `t1→pad`; its key-occurrence (in "t1 t2 pad") makes the real slot `t1→t2`. With
the write-op gate deduping by key, t1 keeps ONE slot = whichever was written LAST,
and the leak-proof shuffle makes that 50/50. So hop2 = P(t1 slot is the real one) ×
P(retrieve) = 0.5 × 0.9 = 45%. hop1 is unaffected (t0 is only ever a key).

So the chaining/retrieval mechanism is sound — each hop is a genuine store lookup —
but the store cannot reliably hold a token that is simultaneously a key and a value.

## Fix (r84)
Don't store pad-valued slots: pad is not a fact, and the spurious `t1→pad` slot is
exactly what clobbers the intermediate. Skipping value==pad writes removes the
collision **order-independently** (no reliance on write order), so t1's only slot is
`t1→t2`. Prediction: hop2 → ~90%, joint 2-hop → ~0.9² ≈ 81%. If it lands with the
broken control at chance, that is CoT-over-memory working (route A).

## Route B, for contrast
r82 (loop-gate latent) was a flat negative; route A here is a mechanism that works
up to a fixable store artifact. Explicit CoT is the viable route.
