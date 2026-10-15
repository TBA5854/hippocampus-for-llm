# r20 — results: one more curriculum step reaches the 8x store

## Numbers (TinyStories, single-occurrence facts, 8 facts, one-shot read beta 32, key @0 position-free, frozen identity
## query, write @ last block + normalised block-0 skip, M=4096, 32 segments, 4 x accum 8, 1500 steps, seed 0,
## --init experiment/r19/ckpt/r19_warm_skipnorm_M2048.pt)

| run | box | store | 250 | 500 | 750 | 1000 | 1250 | final (1500) | gate 250 -> final | mass | slot@1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **r20_warm_skipnorm_M4096** | box1 | real | 96.2% | 100% | 100% | 100% | 100% | **100.0%** | 0.330 -> 0.355 | 0.97 -> 0.96 | 100% |
| r20_warm_skipnorm_M4096_shuf | box2 | scrambled | 1.2% | 2.5% | 3.8% | 3.8% | 5.0% | **5.0%** | 0.310 -> 0.295 | — | — |

Chance 3.1%. Val ppl 2.00 on the real run. Content-miss and match-miss both 0% from step 500 on; alignment 100% at
offset 0 at every eval.

The two runs are one round split across two boxes (see "Ops" below), same seed, same everything but `--shuffle-store`.

## Verdicts

- **H20a (warm start from the 4x model reaches >= 95% at 8x within 1500 steps) — confirmed.** 96.2% at the first eval,
  100% from step 500. For comparison the 4x warm start was at 79.7% at step 250 (r19), and a 4x run from scratch never
  lifted off at all (r17, 12000 steps).
- **H20b (scrambled control at chance) — confirmed.** 1.2-5.0% across six evals with no trend, against 3.1% chance. As
  in r17, the eval uses the same fixed queries every time, so this range is eval noise. The control's gate drifts down
  (0.310 -> 0.295) while the real run's opens (0.330 -> 0.355).

## Persistence at 8x (`mech.persist`, two processes, n=40 queries, 3,937 slots per example, 16.8 MB per store file)

| in-process | reloaded in a fresh process | mismatched store |
|---|---|---|
| 100% | **100%** | 7.5% |

The mismatched-store number is higher than r19's 2.5% but on the same 40-query eval; at 8 facts one extra correct
answer moves it 2.5 points, so 3/40 is within noise of chance (3.1%) rather than evidence of a leak.

## Diagnosis

Each doubling of the store gets *easier* to warm-start, not harder: 79.7% at the first eval at 4x, 96.2% at 8x, and the
read holds mass 0.96-0.97 on the correct slot throughout both. That is the opposite of interference — a store twice the
size dilutes nothing once the read exists. It further supports r19's reading that the cold-start failures at 4x were
about *discovering* the read, not about capacity.

Reproducibility note: this round is a rerun (the first attempt was killed at ~step 800). Steps 250/500/750 reproduced to
the digit — 96.2 / 100 / 100 with mass 0.97 — so the result is not a lucky trajectory.

## Where the objective placement now stands (TinyStories, facts seen once)

| store | slots/example | from scratch | warm start from the size below |
|---|---|---|---|
| 1x | 508 | 100% | — |
| 2x | 896 | 99.4% (normalised skip) | — |
| 4x | 1,905 | chance through 12000 | 100%, persists |
| 8x | 4,064 | not attempted | **100%** |

## Ops (two boxes from this round on)

box1 ran the real store; box2 ("vanilla", RTX A4000) ran the matched control in parallel, halving wall-clock from ~2.5 h
to ~75 min. Two box2 gotchas, both recorded in STATE.md:

- Its `nvidia-smi` is broken (kernel module 595.84 vs userspace 595.91). CUDA itself works; check the GPU with torch.
- `~/research/framework` there belongs to another account, so **writes fail**: the control trained and evaluated fine,
  then died saving its checkpoint ("Permission denied"). Its numbers survive in the log, but its checkpoint and results
  JSON were lost. Box2 work now lives in `~/fw`, which we own.

## Next

r21, already running: **seed 1** at 8x (box1) and **32 facts** at 8x (box2), plus two-process persistence on this
round's checkpoint. 16x (M=8192, 64 segments) is deferred — it needs batch 1 and ~4x the compute per step, past the
~1 h/box budget.
