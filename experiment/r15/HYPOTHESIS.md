# r15 — switch the corpus to TinyStories

## Ideate (from r14)

r14's 4x store reached 97.5% on facts seen once, but the language-model half memorised tinyshakespeare: train LM loss
0.35 vs validation 2.59, validation rising from step ~3000. A 4x-store example draws 16 segments, so 6000 steps is
~390M tokens of a 1.1M-character corpus (~350 passes). Recall survived, but larger stores, longer runs, and any claim
about text need a corpus the model cannot memorise.

`data/tinystories.txt`: 22,474 TinyStories, ASCII-filtered, 20M characters (18x tinyshakespeare), 80-character
vocabulary. Same character tokenizer. (The r14 mid-round beta-64 plan that previously occupied r15 was not needed:
read mass recovered once r14 lifted off.)

## Hypotheses

- **H15a — the task transfers.** 512-slot observation store on TinyStories reaches ~100%, like tinyshakespeare.
- **H15b — overfitting is gone.** On the 4x store, validation LM loss tracks train loss instead of diverging.
- **H15c — the 4x store still holds.** >= 95% on TinyStories.

## Falsifiers

- 512 slots fails on TinyStories -> something corpus-specific (vocabulary, text statistics) interacts with the read;
  check the miss taxonomy before scaling.
- Validation still diverges -> 20M characters is not enough at this token budget; export more.
- 4x recall worse than on tinyshakespeare -> the memorised corpus was helping (e.g. predictable distractor text made
  the gate's job easier).

## Runs (single occurrence, 8 facts, block-0 observation store, one-shot read beta 32, seed 0)

| order | name | M | n_seg | batch x accum | steps |
|---|---|---|---|---|---|
| 1 | r15_ts_M512 | 512 | 4 | 32 x 1 | 4000 |
| 2 | r15_ts_M2048 | 2048 | 16 | 16 x 2 | 6000 |

Batch 16 x 2 (not 8 x 4): r14 used 3.4 GB at batch 8 with the one-shot read.
