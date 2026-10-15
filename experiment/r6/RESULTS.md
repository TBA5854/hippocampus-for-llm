# r6 — results: layer-0 keys replicate; more facts fail on content, and why

## Numbers (key + read @ layer 0, value @ last block, shift 0, identity query; 4000 steps)

| run | facts | seed | best | final | correct-slot reads (final) | verdict |
|---|---|---|---|---|---|---|
| r6_f8_s2 | 8 | 2 | 100.0% | **100.0%** | 100% | third seed at 100% |
| r6_f8_shuf | 8 | 0 | 4.1% | **2.7%** | — | scrambled control at chance |
| r6_f16 | 16 | 0 | 4.5% | **2.2%** | 98% | chance |
| r6_f32 | 32 | 0 | 5.3% | **3.1%** | 100% | chance |
| r6_f8_M2048 | 8 | 0 | — | — | — | OOM at batch 32 -> rerun as r6b at batch 8 |

## Verdicts

- **H6a (replicates) — confirmed.** Seed 2: 100%. Layer-0 keys are 3 of 3 seeds at 8 facts.
- **H6b (control holds for this key space) — confirmed.** Scrambled store 2.7%.
- **H6c (facts scale) — falsified, and not for the reason the hypothesis worried about.** Matching held at
  98-100% at 16 and 32 facts. Recall failed anyway. **The failure moved entirely from matching to content.**
- **H6d (store size scales) — inconclusive: confounded by the batch change.** `r6b_f8_M2048_b8` (M=2048,
  16 segments, 1,920 slots in use, 14 distractor segments, **batch 8**) ended at 5.0% (8 of 160, chance).
  But the cause is the same content failure, not the store: segment-0 loss stalled at **1.93** against a
  0.43 induction floor, from step ~1750 on, and the gate never opened (0.000). Batch 8 sees a quarter of the
  fact segments per step, and at batch 32 the same 8-fact task reached its floor by step ~1000. So this run
  cannot separate store size from the slower induction learning that batch 8 causes.
  **The clean part: correct-slot reads were 100% at every eval across a 1,920-slot store**, weight mass 1.00,
  with two thirds of the added slots being distractor text. Matching does not degrade at 4x store. Recall at
  that size is untested until the value problem is solved, or induction is given the budget to form.

## Diagnosis — the store holds predictions, not observations

The value is the last block's state at the key position with `value_shift 0`, i.e. the model's prediction
"next is `v_j`". So the value carries the answer only if the model could already predict `v_j` from the
fact segment when it wrote. In MQAR that prediction comes from **induction over the tiled repeats**, and
the segment-0 loss measures exactly how much induction there is. Its floor for perfect induction is
(unrepeated first tile / segment) x ln 32:

| facts | repeats per fact | floor with perfect induction | segment-0 loss reached | recall |
|---|---|---|---|---|
| 8 | 8 | 16/128 x 3.47 = **0.43** | **0.41** (at floor) | **100%** |
| 16 | 4 | 32/128 x 3.47 = **0.87** | **3.29** (none) | chance |
| 32 | 2 | 64/128 x 3.47 = **1.73** | **2.61** (partial, from step ~2250) | chance |

Recall is high exactly where induction reached its floor and at chance where it did not, while correct-slot
reads stay ~100% throughout. The read finds the right slot; the slot has nothing in it yet.

`r6_f32` is the telling one: induction began forming at step ~2250 (loss 3.36 -> 2.61) and the budget ended
before it finished, with the gate still shut. The order is the predicted one — value content first, then
the gate opens, then recall.

**Consequence.** A fact seen *once* can never be predicted from context at write time, so in this
configuration it can never be stored. The 8-fact 100% is a real, controlled result for in-block storage
and persistence, but it depends on the fact being inferable in the writing segment. That is the tightest
limit on the objective, recorded in `experiment/OBJECTIVE.md`. Not yet tested directly with a
single-occurrence-fact task.

**Fix direction** (a new objective, not launched): the value has to carry what was *observed*, not what was
predicted — take the value from a layer that still holds token identity at the position of `v_j`
(shift 1), keeping the key at layer 0 where it does not drift.

## Also measured

- Memory for a store-carrying example scales with **n_seg x M**: every segment's `[B, S, M]` settle
  intermediates stay alive until backward. 16 segments x a 4x store at batch 32 needed > 15 GB from one
  process. Day-scale stores need either smaller batches, truncated backprop across segments, or a store
  read that does not materialise `[B, S, M]`.
