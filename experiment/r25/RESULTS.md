# r25 — results: the fact count was never a capacity limit either

## Numbers (1x store, 128-symbol vocabulary, box2, seed 0; each run warm-started from the previous checkpoint)

| run | facts | start | 250 | 500 | 1000 | 2500 | final (4000) | gate | mass |
|---|---|---|---|---|---|---|---|---|---|
| **r25_1x_f16_warm** | 16 | from the 8-fact model | 88.4% | 90.6% | 98.9% | 100% | **100.0%** | 0.153 -> 0.318 | 0.99 |
| **r25_1x_f48_warm** | 48 | from the 16-fact model | 99.7% | 98.4% | 99.4% | 100% | **100.0%** | 0.320 -> 0.392 | 0.99 -> 1.00 |
| r25_1x_f48_warm_shuf | 48 | same, scrambled store | 0.6% | 0.8% | 0.8% | 0.6% | **1.1%** | 0.313 -> 0.250 | — |

Chance 0.8%. Both warm runs: slot@1 100%, content-miss and match-miss 0% from step 2500 on.

Against the cold runs of the *same* configurations (r23, r24, 8000 steps — twice the budget):

| facts | cold | warm | cold gate | warm gate |
|---|---|---|---|---|
| 16 | 3.8% | **100%** | 0.057, frozen | 0.318, opens |
| 48 | 0.9% | **100%** | 0.008, shut | 0.392, opens |

## Verdicts

- **H25a (16 facts warm-started >= 95%) — confirmed.** 88.4% at the first eval, 100% by 2500.
- **H25b (the curriculum continues to 48 facts) — confirmed.** 99.7% at the first eval.
- **H25c (scrambled control at chance) — confirmed.** 1.1% final against 0.8% chance, never above 1.6% at any
  eval, with the inherited gate closing steadily (0.313 -> 0.250) as training learns a scrambled store is useless.

## Diagnosis — one failure mode, three costumes

Three axes looked like independent ceilings. All three behave identically:

| axis | cold | warm-started | what actually failed |
|---|---|---|---|
| store size 1x -> 16x | 4x at chance through 12000 steps (r17) | 100% at every size (r19, r20, r22) | lift-off |
| fact count 8 -> 48 | 16 facts 3.8%, 48 facts 0.9% (r23, r24) | **100% at both** (this round) | lift-off |
| key vocabulary 32 -> 128 | works cold, 3x slower to lift off (r23) | — | lift-off *cost*, not a ceiling |

In every failing run, at every eval, **slot@1 was 100%**. Retrieval was never the problem: position-free layer-0 keys
with a frozen identity query make matching close to exact token lookup, and it simply works. What has to be *learned*
is the use of what was retrieved — the gate opening and the downstream blocks decoding the injected vector.

That is a bootstrapping deadlock. The gate gets useful gradient only if the injection helps the loss; the injection
helps only if downstream weights can already decode it; neither exists at initialisation. The measurements show the
deadlock resolving in whichever direction it starts:

- **Cold, hard task:** gate shuts (0.008), read mass decays (0.97 -> 0.70 at 48 facts). The decaying mass is a
  *consequence* — with the gate shut there is no pressure keeping key representations separated.
- **Warm:** the decoder already exists, the first injection is immediately useful, the gate opens monotonically
  (0.15 -> 0.39) and mass stays at 0.99-1.00.

So the project's ceilings were one mechanism seen three times. What a curriculum buys is not capacity but an escape
from the deadlock, and once escaped it transfers across store size, fact count and vocabulary alike.

## What this does not show

- The curriculum is a *training-time* device. It says nothing about adding facts at inference on a frozen model —
  that is the store's job, and it is what sleep/consolidation is for.
- 48 facts is where the task's own geometry runs out at `seg_len=128` (48 pairs = 96 of 128 tokens in segment 0), not
  where the memory does.
- All warm runs here are seed 0 at the 1x store.

## Next

r26: **can a stored value be decoded back to a token id?** (`mech/decode.py`, written this round.) This is the
prerequisite for sleep as an eviction mechanism: `wake_sleep.py` already consolidates facts into weights with reads
disabled, but its replay is reconstructed from a known persona rather than read out of the store — its own docstring
names that limitation, which existed because the old pooled store carried no token identity. The current per-token
store writes the *observed* token with a block-0 skip, so the identity should survive. If it decodes, sleep can replay
from the store and free the slots it consolidates: eviction becomes "consolidate, verify with reads off, then free",
and an episode's completeness stops being a heuristic (a closed episode has a replay target; an open one does not).
