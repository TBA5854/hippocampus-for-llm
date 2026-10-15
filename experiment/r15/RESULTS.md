# r15 — results: on TinyStories the 4x store reaches 100%, and the language model no longer memorises

## Numbers (single-occurrence facts, 8 facts, block-0 observation store, one-shot read beta 32, seed 0)

Corpus: `data/tinystories.txt` (22,474 stories, 20M ASCII characters).

| run | M | slots filled | batch x accum | steps | first >= 50% | first >= 95% | first 100% | final | train / val LM loss | read mass min / final |
|---|---|---|---|---|---|---|---|---|---|---|
| r15_ts_M512 | 512 | 384 | 32 x 1 | 4000 | 1250 | 1500 | 1750 | **100.0%** | 0.72 / 0.75 | 0.98 / 0.98 |
| **r15_ts_M2048** | 2048 | 1,920 | 16 x 2 | 6000 | 3750 | 4500 | 5500 | **100.0%** | 0.70 / 0.73 | 0.84 / 0.97 |

Correct-slot (top-1) reads 100% at every eval of both runs. Largest validation-minus-train LM loss over the second
half of each run: +0.04 (512 slots), +0.09 (4x).

Against tinyshakespeare (r14), same store sizes:

| | tinyshakespeare | TinyStories |
|---|---|---|
| 512 slots | 100%, lift-off ~750 | 100%, lift-off ~1250 |
| 4x store | 97.5% final, train 0.35 / val **2.59** | **100%** final, train 0.70 / val **0.73** |

## Verdicts

- **H15a (the task transfers) — confirmed.** 100% at 512 slots. Lift-off ~500 steps later than on tinyshakespeare.
- **H15b (overfitting is gone) — confirmed.** Validation LM loss tracks train within 0.1 for the whole 4x run (r14: a
  gap of 2.2 and widening).
- **H15c (the 4x store still holds) — confirmed and improved.** 100% final (r14 97.5%), and past r14 at matching steps
  from 3750 onward. The memorised corpus was not helping; if anything it was costing the last few points.

## Store-size scaling, final picture (observation store, facts seen once, effective batch 32)

| store | slots filled | corpus | first >= 50% | final |
|---|---|---|---|---|
| 1x | 384 | tinyshakespeare / TinyStories | ~750 / ~1250 | 100% / 100% |
| 2x | 896 | tinyshakespeare | ~2500 | 100% |
| 4x | 1,920 | tinyshakespeare / TinyStories | ~3500 / ~3750 | 97.5% / **100%** |

Store size moves lift-off, not the ceiling. The pre-lift-off dip in read mass (to ~0.81-0.84, recovering to ~0.97)
reproduced on the new corpus, so it is a property of the read at scale, not of the data.

## Default going forward

TinyStories is the corpus for all further rounds.

## Persistence at the 4x store (offline, two processes, on r15_ts_M2048's checkpoint)

| | recall |
|---|---|
| in-process | **100.0%** |
| store saved, loaded in a fresh process, query segment has no facts | **100.0%** |
| fresh process, mismatched store file | 2.5% |

n = 40 queries (batch 8 x 5): kept small because a 4x store is 4.2 MB per example (2,048 slots x 256 x 4 bytes x 2,
1,905 filled). Small sample; the result matches every larger persistence test so far. Files in `r15/offline/`.
