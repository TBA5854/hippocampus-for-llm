# r35 — results: the store answers in the text vocabulary, partially. First capability result on language.

PG-19 prose as distractors, 8 facts stated as `qx:mk` in segment 0, queried as `?qx:` at the end of segment 3.
Segment 0 is long out of context by then, so a correct answer can only come from the store. Char level, seed 0,
6000 steps, r23's BASE verbatim (see HYPOTHESIS.md).

## The numbers

| run | exact | char | answer CE | gate | final val |
|---|---|---|---|---|---|
| `r35_ceiling` — facts pasted into the query segment | **97.2%** | 98.6% | 0.054 | 0.293 | 1.331 |
| **`r35_on` — memory, facts OUT of context** | **10.0%** | **47.1%** | **1.736** | 0.169 | 3.501 |
| `r35_shuf` — scrambled store, matched | 0.2% | 3.8% | 3.259 | 0.034 | 5.338 |
| `r35_off` — no memory | 0.2% | 4.2% | 3.253 | — | 5.324 |
| **chance** | **0.15%** | **3.8%** | **3.258** | | |

Best and final are identical for every run (no peak-then-decay), so there is nothing hidden in the trajectory.

## Verdicts

- **H35a (facts stated in text are recalled from the store above chance) — confirmed.** 10.0% exact against 0.15%
  chance is **67x**, and against the matched scrambled control's 0.2% it is **50x**. Character accuracy 47.1% against
  3.8%, and answer CE 1.736 against 3.258. All three metrics agree, which is what makes this a result rather than a
  fluctuation.
- **H35c (both controls cannot answer) — confirmed, unusually cleanly.** The scrambled store lands on **3.8% char
  and CE 3.259** against a chance reference of 3.8% and log(26) = 3.258 — three digits, on two independent metrics.
  The no-memory floor is the same. And the scrambled run **closes its gate to 0.034** while the real one holds
  0.169: the model learns a randomised store is not worth reading. The ceiling at 97.2% proves the task is learnable
  and the instrument is sound, so the gap below is about the store, not about the task.
- **H35b (the 2-character metric understates a partial binding) — confirmed, and it matters.** Exact match says
  10.0%; character accuracy says **47.1%**. Roughly half the answer characters are right while only a tenth of the
  *pairs* are. Reporting exact match alone would have described this round as near-failure.

## What this establishes, and what it does not

**Establishes: the mechanism is not about symbol tokens.** Every prior recall result (r1-r33) used MQAR keys and
values drawn from a vocabulary *outside* the text, so they could never collide with prose. Here a key is a letter
bigram in the same character space as the books, competing with English bigrams the model already predicts
confidently, and the store still carries it 67x above chance with both controls pinned at chance. That was the open
question the audit exposed — 26 rounds of MQAR, one of LM, zero of facts-in-text — and it is answered.

**Does not establish: a whole binding.** 10.0% from the store against **97.2%** with the facts in context is the
round's real limitation. The model can do this task nearly perfectly when it can see the facts; the store delivers
part of the association and not the rest. Character accuracy at 47.1% locates it: the answer is two characters and
the store reliably supplies about one of them.

> **CORRECTED BY r36 — the paragraph below was wrong.** It attributed the gap to "matching intact, content partial"
> by analogy with r9, and flagged that as unmeasured. r36 measured it: per-position accuracy is **pos0 11.4% /
> pos1 80.5%**, so the *second* character is the one that works and the first sits on **1/n_facts = 12.5%**. The
> cause is **query ambiguity** — the first answer character must be retrieved by reading at the `:` token, which
> appears once per fact and so matches all eight fact slots equally. It is not content loss and not chained
> retrieval. See [r36/RESULTS.md](../r36/RESULTS.md). The original text is kept below.

That shape — matching intact, content partial — is the failure this project has met before and named. r9 measured
100% matching with content lost between store and output; the "Mittens for Miso" fusion result in the report is the
same thing. The diagnostic that separates matching from content (`slot_diagnostic`) is **MQAR-only** and did not run
here, so the attribution is by analogy rather than measured. **That is the obvious r36**: extend the diagnostic to
the recall task and find out whether the read lands on the right slot and loses the second character, or lands
between slots.

## Why the LM loss is so different across arms, and why it does not matter

Final val is 1.331 (ceiling) / 3.501 (memory) / 5.33 (both controls). This is **not** a language-modelling
comparison: `--answer-weight 5.0` puts five times the weight on two answer characters, and the arms that can answer
drive that term to near zero while the arms that cannot leave it at chance. The LM term is the same in all four.
Perplexity is not the metric for this round and is reported only for completeness.

## What this does NOT show

- **One seed.** r34 in this same session showed a 3.3x magnitude spread across four seeds on a smaller effect, and
  r32 measured 72 points of spread on MQAR. 10.0% is one draw; the spread is unmeasured.
- **Tiled facts.** `make_batch` repeats the fact string through segment 0, so each fact appears many times. MQAR's
  harder single-occurrence setting (`--no-tile`) is an mqar-only flag and does not apply to this task; the
  single-occurrence version of this round is unbuilt.
- **8 facts, 512 slots, 4 segments, one store size.** None of the axes r21-r25 pushed on MQAR (fact count, store
  size, key vocabulary) has been touched here, and the curriculum that unlocked all of them is untried.
- **No consolidation.** Whether these text-vocabulary bindings survive sleep, or re-encode after it, is untested —
  r26-r33 are all MQAR.
- The answer is still two characters of a 26-letter alphabet, which is a synthetic binding *expressed* in text, not
  a natural-language fact like "Miso's cat is Mittens". The next rung up is word-level answers.
