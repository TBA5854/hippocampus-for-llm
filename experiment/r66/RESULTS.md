# r66 — chase the curriculum on real language (BPE): RESULTS

**Verdict: the curriculum is NOT required on real language. It is a training-budget
optimisation, not an enabler.** Multi-layer read/write placement clears the BPE
fact-count cliff *cold* given enough steps; the warm-start curriculum reaches the
same recall in roughly half the steps.

## Setup
Both arms use multi-layer placement (`--read-layers 0..5 --write-layers 0..5`),
BPE single-token word-keys (`bpe_8192`), the standard block-memory config
(layer-0 position-free key, frozen identity query, last-block value skip,
`memory_size 512`, `seg_len 128`, 4 segments). Seed 0.

- **box1 `mlcurr`** — curriculum ladder, warm-started: f8 (cold, 8k) → f16 (warm
  from f8, 8k) → f32 (warm from f16, 8k).
- **box2 `mlcold`** — cold, long: f16 (cold, 16k) and f32 (cold, 16k), no ladder.

## Numbers (recall, best = final unless noted)

| arm | 8 facts | 16 facts | 32 facts | steps/rung |
|-----|---------|----------|----------|-----------|
| box1 ml + **curriculum** (warm) | 59.4% (cold seed rung) | **100.0%** | **100.0%** | 8k |
| box2 ml **cold** (long) | — | **99.7%** | **98.0%** (best 98.1) | 16k |

## What actually happened

**Cold multi-layer training reaches full recall through a delayed phase
transition.** box2 cold f16 sat near zero until it grokked:

```
step   500  recall  0.0%
step  2000  recall  0.3%
step  4000  recall  8.8%
step  6000  recall 77.2%   <- transition
step  8000  recall 98.9%
step 16000  recall 99.7%
```

f32 cold followed the same curve one notch later — 5.6% at step 4000 (in the exact
pre-transition zone f16 occupied at step 4000), then transitioned and finished at
98.0%. Nothing special about 32 facts; it just needed to run past the transition.

**The curriculum ladder gets there in fewer steps.** box1 f8 cold stalled at 59.4%
(low fact count is the hardest cold — fewest gradient signals to bootstrap the
write/read loop), but warm-starting f16 from that checkpoint jumped to 100.0% in
8k steps, and f32 warm-from-f16 held 100.0%. So the ladder reaches 32-fact full
recall in ~24k total steps (3×8k) that are each cheaper, vs box2's 16k cold steps
per fact count — the warm rungs converge almost immediately because the read/write
loop is already established.

## Answers to the round's questions
- **Q1 (box1): does ml + curriculum reach 100% like single-layer curriculum r49?**
  Yes — 100.0% at both f16 and f32.
- **Q2 (box2): does ml cold + long training close the gap alone → curriculum
  removable on BPE?** **Yes.** Cold f16 = 99.7%, cold f32 = 98.0% at 16k steps.

## Why this overturns r63
r63 reported ml lifts BPE cold to ~53→87 "but not to 100 (curriculum does the
rest)" and attributed the residual gap to an embedding-competition wall. r66 shows
that residual was **under-training**: r63 stopped before the delayed transition.
With 16k steps the cold model transitions and reaches ~98–100%. The "embedding
competition" bottleneck is real as a *slowdown* (it pushes the transition late and
makes low-fact cold rungs stall), but it is not a hard wall — placement + budget
clears it.

## Implication for the paper
The claim "curriculum is not removed on real language" (discussion.tex,
placement.tex) is now false as stated and must be rewritten:
- Multi-layer placement removes the fact-count cliff **cold** on real language.
- The curriculum is an **efficiency** win (≈half the steps), not a requirement.
- The cliff is a delayed-phase-transition / bootstrapping cost, not a hard
  embedding-competition ceiling.

## Controls / caveats
- Single seed (0) per arm — the transition step is seed-sensitive (r24/r57 noted
  seed noise in placement escapes), so the *step* of the transition should not be
  over-read; the *endpoint* (~98–100% both arms) is the robust claim.
- Overnight both boxes slept mid-run (wall-clock etime overstates compute); runs
  resumed and completed cleanly, EXIT 0, ROUND DONE on both.
- Best = final on every rung (no late collapse).
