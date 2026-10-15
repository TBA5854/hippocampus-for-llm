# r48 — BPE + curriculum reaches 100% at 8 facts, and the fact count scales past the char cap

## Ladder (box1): the curriculum takes BPE to the ceiling

| run | exact | answer CE | gate |
|---|---|---|---|
| `r48_bpe4` — 4 facts cold | 37.2% | 4.078 | 0.236 |
| **`r48_bpe8_curr` — 8 facts warm from 4f** | **100.0%** | 0.034 | 0.881 |

For reference: BPE 8 cold (r47) 49.8%, facts-in-context ceiling 96.1%, char-level 8-fact best 79% (r44).

## Scale (box2): past the char task's 13-fact cap

| run | exact | answer CE | gate |
|---|---|---|---|
| `r48_bpe16` — 16 facts cold | 52.8% | 4.808 | 0.259 |
| `r48_bpe32` — 32 facts cold | 5.9% | 6.598 | 0.143 |

Chance ~1/1899 = 0.05%. The char task cannot represent 16 or 32 facts at all (2 letters/key from 26 caps it at 13).

## Verdicts

- **H48a — the curriculum takes BPE 8 facts to the ceiling. Confirmed, fully.** 100.0% exact, CE 0.034, gate 0.88.
  This is the objective met on real language: 8 facts stated in English, out of context, answered from the store,
  perfectly. It clears char-level's ~80% asymptote (r44) decisively — single-token addressing has no residual
  per-character tax to cap it, so once bootstrapping is escaped it goes all the way.
- **H48b — BPE scales past the char cap. Confirmed.** 16 facts at **52.8% cold** (impossible in the char task) and
  32 facts at **5.9% cold** — both far above chance. Cold recall falls with fact count, as expected from the
  bootstrapping axis (r25), but the vocabulary no longer bounds the fact count, and the curriculum that took 8 facts
  from 49.8% to 100% should take these up too (r49).

## The one anomaly, recorded

BPE **4 facts cold (37.2%) is LOWER than 8 facts cold (49.8%, r47)** and far below char 4-fact cold (90.3%). The
likely cause is output-vocabulary size: the BPE value is one of ~1899 word-tokens, so the readout has a far larger
space to learn cold than the char task's 26 letters, and at very low fact counts that cost dominates. It did not
matter for the ladder — 8 facts warm from that same 37.2% base still reached 100% — so the curriculum is robust to a
weak base, but the cold low-fact regime is a genuinely harder readout for BPE and is worth noting rather than
smoothing over.

## What this establishes

The two language bottlenecks are now both beaten, on real-language facts:
- addressing tax (multi-token key) — removed by BPE (r47)
- bootstrapping deadlock — escaped by the curriculum (this round, to 100%)

And the fact count is no longer vocabulary-bound. The objective's ">= 8 facts, out of context, from the store, with
controls at chance" is met at **100%** with single-token real-word facts (controls: r47 floor 0%, scrambled 0%).

## What this does NOT show

- 16/32 facts with a curriculum ladder (r49) — cold only here.
- One seed per point. The 4-fact cold anomaly especially wants a second seed.
- Persistence across a process restart on the BPE task (the objective's save/reload demo) — done on MQAR, not yet on
  BPE recall.
