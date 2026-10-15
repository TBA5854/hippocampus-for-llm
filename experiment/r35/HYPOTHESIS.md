# r35 — language: facts stated in text, queried in text, on PG-19 prose

> Supersedes the earlier r35 placement plan, which moves to r36. The user asked for language *and* PG-19:
> this round is both — the recall task with PG-19 books as the prose.

## The gap this closes

Audited, not remembered: **26 rounds used `--task mqar`, one used `--task lm` (r34), and `--task recall` has never
been run.** The tinyshakespeare -> TinyStories switch at r15 looks like a move toward language but was not — those
corpora were only the *distractor text*. The facts were always MQAR symbol pairs drawn from a dedicated symbol
vocabulary, the query was always a symbol, and the metric was always exact match on one token against 1/32 or 1/128
chance.

So the project's position is:

| | task | metric | result |
|---|---|---|---|
| r1-r33 | MQAR symbols | exact match, 1 token | **100%**, 3 seeds, 48 facts, 16x store, persisted |
| r34 | PG-19 next-char | perplexity shape | +0.8% on seed 0, **not reproduced on seed 1** |
| **r35** | **facts in text** | **exact match, chars** | this round |

The objective says the model "answers from its own store with nothing in the prompt". That is demonstrated **on
symbols only**. r34 measured perplexity, which is not a capability. This round asks for the capability in the text
vocabulary.

## What makes it harder than MQAR, which is the point

MQAR keys and values are tokens *outside* the text vocabulary, so they cannot collide with prose. Here a key is a
letter bigram like `qx` written as `qx:mk`, drawn from the **same character vocabulary as the books**. The store,
the read, and the LM distribution all share one space, so:

- the key competes with ordinary English bigrams the model already predicts confidently
- the answer is **two characters**, not one token
- the query `?qx:` sits at the end of a segment of real prose

## The trap this round must not fall into, stated before it runs

The note that created MQAR says exactly why `recall` was abandoned:

> "Character-level made exact-match a poor instrument: the answer was TWO characters, so a real but small retrieval
> edge (measured: CE 3.148 vs a 3.256 scrambled control, log(26)=3.258) almost never produced a correct pair."

**Exact match read ~0 while the model genuinely knew something.** Running this round on exact match alone would
reproduce that and mis-report it as "the store cannot carry language". So three metrics are logged, and they fail
independently ([`mech/tasks.py:answer_stats`](../../mech/tasks.py), five tests):

| metric | chance | what it catches |
|---|---|---|
| `exact` | (1/26)^2 = 0.15% | both characters. **The capability claim.** |
| `char_acc` | 1/26 = 3.8% | a half-learned binding that exact match rounds to zero |
| `answer_ce` | log(26) = 3.258 | an edge too small to move any argmax |

**A positive needs `exact`. A negative needs all three**, or it cannot separate "the store does not carry this" from
"the ruler is too coarse to see it". That distinction is the whole reason this round is worth running rather than
assuming MQAR settles it.

## Architecture: the r23 BASE verbatim

r34 cost a full set of runs by taking the placement from prose. The flags here are copied from
[`experiment/r23/run.sh`](../r23/run.sh), a round that reached 99.7%, changing only the task, the corpus and the
sizes. In particular it keeps what r34's "corrected" config still lacked:

- `--value-skip-layer 0 --value-skip-norm` — the **normalised block-0 skip**. r17/r18 established the last-block
  write does not work without it (plain skip: 1x 100%, 2x 28.1%; normalised: 2x 99.4%). r34 had no skip path at all.
- `--readout graded`, `--beta 32 --beta-max 64 --beta-grad 4 --n-steps 0`, `--write-threshold 0.6`
- `--answer-weight 5.0` — the answer is 2 characters in 128 positions, ~1.5% of the gradient. Without this the
  model ignores it; the task docstring records a **6.9% ceiling even with the facts in context**.

Store sizing: `seg_len 128`, `n_seg 4` -> 4 x 127 = 508 slots into `--memory-size 512`. Full, no overflow, matching
the clean-headline sizing used since r1.

## Hypotheses

- **H35a — facts stated in text are recalled from the store above chance on `exact`.** If MQAR's 100% is about the
  mechanism rather than about symbol tokens, this should clear 0.15% chance by a wide margin. **Falsified if** `exact`
  sits at chance *while* `char_acc` and `answer_ce` also sit at chance — that is a real negative and would say the
  binding does not survive sharing a vocabulary with the prose.
- **H35b — the blunt-instrument reading is distinguishable.** If `exact` is near zero but `char_acc` > 3.8% and
  `answer_ce` < 3.258, the store carries a *partial* binding and the 2-character metric is hiding it. That is the
  pre-r1 result reproduced on a much better architecture, and it would say the honest next step is a single-token
  language answer (word-level), not more training.
- **H35c — `hc off` cannot answer and `--shuffle-store` cannot answer.** Both at chance on all three metrics. If
  either clears chance the task leaks and every number is void. `--facts-in-context` is the ceiling: if *it* cannot
  answer, the model cannot do the task at all and the store is not what is being measured.

## Runs

box1 gets the pair that decides the round; box2 gets the ceiling and the control.

```
BASE = r23's, with --task recall --data-books data/pg19 --n-facts 8
       --seg-len 128 --n-seg 4 --memory-size 512 --batch 32 --steps 6000

box1   r35_on        --hc on                    the number
       r35_off       --hc off                   floor
box2   r35_ceiling   --hc on --facts-in-context  ceiling
       r35_shuf      --hc on --shuffle-store     matched control
```

PG-19 books supply the distractor prose. Book boundaries are irrelevant here — prose is independent per-segment
filler, not something the store must bridge — so the corpus is flattened for this task, which
[`BookCorpus.flat()`](../../mech/data.py) does explicitly rather than by accident.

## What would make this round wrong

- Reading a null off exact match alone. Guarded above.
- The ceiling failing, which would make every other number meaningless. Checked first.
- Quoting a cold-start null as a capacity result: r25 established that store size, fact count and key vocabulary
  **all** fail cold for one reason (the gate/decoder bootstrapping deadlock) and a curriculum escapes all four. If
  `r35_on` is at chance with a shut gate, the next step is a warm start from an MQAR checkpoint or from a tiled/
  fewer-facts rung — not a conclusion about language.
