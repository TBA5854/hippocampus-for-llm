# r37 — results: the 1/n_facts ceiling is a CLIFF, not a curve, and a shift addresses around it

| run | exact | char | **pos0** | 1/n_facts | pos1 | answer CE | gate |
|---|---|---|---|---|---|---|---|
| **`r37_f4`** — 4 facts | **90.3%** | 95.1% | **94.1%** | *25.0%* | 96.1% | 0.160 | 0.275 |
| `r36_lang_s1` — 8 facts | 10.3% | 45.9% | 11.4% | *12.5%* | 80.5% | 1.734 | 0.206 |
| `r37_f12` — 12 facts | 6.6% | 42.6% | 7.8% | *8.3%* | 77.3% | 1.919 | 0.147 |
| **`r37_shift3`** — 8 facts, `--value-shift 3` | **31.6%** | 55.1% | **62.0%** | — | 48.1% | 1.429 | 0.130 |

All seed 1, 6000 steps, otherwise identical. Chance: exact 0.15%, char/pos 3.8%, CE 3.258.

## H37a — confirmed at 8 and 12, and falsified at 4. The shape is a cliff.

`pos0` tracks 1/n_facts closely at the two higher counts — **11.4% against 12.5%**, **7.8% against 8.3%** — which is
the r36 diagnosis holding on two independent points. Then at 4 facts it is **94.1%, not 25%**, with exact match at
**90.3%**.

So the delimiter ambiguity is **not a hard combinatorial ceiling**. The model *can* disambiguate eight identical
`:` keys in principle; at 4 facts it does so almost perfectly. What the round actually found is a **cliff between 4
and 8 facts**, with the 1/n_facts line describing where the model sits *after* it has fallen off — a model that has
not learned to disambiguate defaults to a uniform draw, and a uniform draw over n facts is 1/n. The line is the
signature of the failure, not its cause.

**This is r24's cliff, in the text vocabulary.** r24 found MQAR breaks between 8 and 16 facts (99.7% -> 3.8%) with
`slot@1` at 100% throughout, i.e. never a matching failure. Here the same cliff sits between 4 and 8, lower because
the task is harder: the key is a bigram sharing a vocabulary with English prose rather than a distinct symbol.

And r25 established what that cliff is: **not a capacity limit**. Store size, fact count and key vocabulary all fail
cold for one reason — the gate/decoder bootstrapping deadlock — and a curriculum escapes all of them (16 facts
3.8% -> 100%, 48 facts 0.9% -> 100%). The gate numbers here are consistent with that reading: 0.275 at 4 facts,
0.206 at 8, 0.147 at 12 — the gate opens less the further past the cliff the run starts.

**The prediction that follows is sharp and cheap:** warm-starting 8 facts from the 4-fact checkpoint should break the
1/n_facts line. That is r38, and it re-uses the project's central result rather than inventing a new mechanism.

## H37b — confirmed. Addressing around the delimiter triples exact match.

`--value-shift 3` moves every retrieval off `:` and onto the key's own characters, which the task draws distinct:
the slot keyed on `a` returns `c`, the slot keyed on `b` returns `d`.

- **`pos0` 11.4% -> 62.0%**, decisively through the 12.5% line. The first answer character is no longer a uniform
  draw, which is the direct confirmation of r36's diagnosis: change *what token the retrieval keys on* and the
  ceiling disappears.
- **exact 10.3% -> 31.6%**, a 3x gain.
- **`pos1` 80.5% -> 48.1%**, which the hypothesis pre-registered as a possible cost and which happened: at shift 3
  the slot keyed on `c` returns the character three later, not `d`, so the second character loses the clean route it
  had at shift 1.

So a single shift cannot serve both positions — each shift makes one hop easy and the other awkward. 31.6% is the
best of the two arrangements, not the ceiling of the idea; a layout where **both** answer characters are addressed
by unique tokens at the same shift is untested and is not obviously impossible.

**This is a task-layout fix, not an architecture fix**, and it should not be read as one. It works by arranging for
retrieval to key on a unique token. r36's constraint is untouched and still holds:

> A position-free single-token key can only address a single distinct token. Making a fact retrievable means giving
> it a unique token to be retrieved by.

## What this does NOT show

- **The 4-fact result has no matched control.** 90.3% is far above anything else here, and `r35_shuf` was run at 8
  facts. A 4-fact scrambled control is the first run of r38 — until it lands, 90.3% is uncontrolled.
- **One seed per point.** The sweep is 4/8/12 at seed 1 only. r36 showed this task replicates tightly (1.2-point
  spread over three seeds at 8 facts), which is reassuring but not the same as measuring it here.
- **The cliff is located between 4 and 8, not resolved.** 5, 6, 7 are unmeasured, so "cliff" describes a 2x gap in
  fact count, not a knife edge.
- **The curriculum prediction is untested.** It is a prediction from r25, not a result.
- Still 8 facts, 512 slots, tiled, char level, no consolidation.
