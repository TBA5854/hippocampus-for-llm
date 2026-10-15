# r36 — results: r35's ceiling is key ambiguity, and it is exactly 1/n_facts

## box1 — where the answer is lost

| run | exact | char | **pos0** | **pos1** | answer CE | gate |
|---|---|---|---|---|---|---|
| `r35_on` (seed 0) | 10.0% | 47.1% | *not measured* | *not measured* | 1.736 | 0.169 |
| `r36_lang_s1` | **10.3%** | 45.9% | **11.4%** | **80.5%** | 1.734 | 0.206 |
| `r36_lang_s2` | **9.1%** | 48.9% | **11.9%** | **85.9%** | 1.624 | 0.159 |
| chance | 0.15% | 3.8% | 3.8% | 3.8% | 3.258 | |
| **1/n_facts** | | | **12.5%** | | | |

*(r35 ran before the per-position metric existed. Its row reads "not measured", not zero.)*

### H36a — falsified, on its third branch, and the third branch has a mechanism

The prediction was `pos0` high and `pos1` low: find the first answer character, fail to chain to the second. The
measurement is the **opposite and lopsided** — `pos0` 11.4% / 11.9%, `pos1` 80.5% / 85.9%.

`pos0` sits on **1/n_facts = 12.5%**, and that is not a coincidence:

> A position-free layer-0 key **is the token embedding**. To predict the first answer character the model must read
> at the **`:`** token. But `:` appears **once per fact — eight times in segment 0** — so every one of the eight colon
> slots matches the query equally well. The read is a **uniform draw over the fact set**, and 1/8 = 12.5% is what it
> scores.
>
> The second character is keyed on the **first value character**, which the task draws from a permutation and is
> therefore unique. That retrieves at 80-86%.

The arithmetic closes: `pos0 x pos1` = 0.114 x 0.805 = **9.2%** and 0.119 x 0.859 = **10.2%**, against measured exact
of 10.3% and 9.1%. So the two hops are near-independent and **the whole of r35's 10% is (1/8) x (the second hop
working)**.

### This corrects r35's diagnosis, which was wrong

r35/RESULTS.md attributed the 10%-vs-97.2% gap to "matching intact, content partial — r9's failure and the report's
'Mittens for Miso' fusion, again", and flagged it as *by analogy, not measured*. The measurement says otherwise:

- it is **not** content loss — when the read is unambiguous the content arrives 80-86% of the time
- it is **not** chained retrieval failing — the *second* hop is the one that works
- it **is matching**, but not the correlated-key kind r5 met: the key is perfectly sharp, it is the **query** that is
  ambiguous, because the only token available at the retrieval point is shared by every fact

That distinction matters because the two have opposite fixes. Correlated keys are fixed by centring or sharper
retrieval (r5). An ambiguous query is not fixable by any amount of retrieval sharpness — **there is no information
in the query to sharpen toward.**

### The constraint this exposes

MQAR never met this because its query *is* the key token and its key symbols are distinct by construction. Written
in text, a fact needs a multi-character key (`ab`), and the retrieval point sits at the delimiter:

> **A position-free single-token key cannot address a multi-token key.** The property that makes retrieval exact —
> the key being the token embedding itself — also means an addressable key must be **one distinct token**. This is
> the same trade-off r31d found from the other side: project the key to make it smaller and exactness dies; the
> exactness is load-bearing, and its price is single-token addressing.

That is a real limit on "facts in text", and it was invisible while the task was MQAR.

### H36b — confirmed. This effect replicates where r34's did not

10.0% / 10.3% / 9.1% across three seeds, `char` 47.1 / 45.9 / 48.9, CE 1.736 / 1.734 / 1.624. A **1.2-point** spread
on exact match. r34's PG-19 effect varied 3.3x across four seeds; this one does not vary, because it is pinned by a
combinatorial ceiling (1/n_facts) rather than sitting in training noise.

## box2 — the empty-store read: H36c FALSIFIED

r34 found the memory model loses loss at segment 0, where the store is empty but the gate is open, and framed it as
"an open gate on an empty store costs". `--suppress-empty-read` skips retrieval while `n_stored == 0`. Prediction:
the s0 cost goes to ~zero and net benefit rises to meet the retrieval gain. Scored against each seed's own r34
control.

| seed | s0 gap: empty-read -> suppressed | net benefit: empty-read -> suppressed |
|---|---|---|
| 0 | +0.0006 -> +0.0007 | +0.0078 -> +0.0088 |
| 1 | -0.0081 -> **-0.0065** | +0.0006 -> +0.0016 |

**Falsified.** On seed 1 — the seed that actually paid the s0 cost — suppressing the empty read moved it only
-0.0081 -> -0.0065, well within the ~0.003 eval jitter. On seed 0 it was near zero already and stayed there. Net
benefit rose by ~0.001 on both, also within jitter.

**Diagnosis — the s0 cost is model difference, not the empty read.** The memory-on and memory-off models are
different trained models; at segment 0 the memory model is slightly worse for reasons that have nothing to do with
what it reads there (it spent capacity/trajectory on the memory pathway). Suppressing the read cannot fix a cost the
read never caused. r34's "open gate on an empty store costs loss" framing was wrong, and this is the correction:
segment 0 differences between the two models are not attributable to the empty read.

This does not change r34's headline — the *later-segment* retrieval gain and the scrambled-control collapse stand.
It removes one over-interpreted sub-finding.


## What this does NOT show

- **The 1/n_facts explanation is inferred from one fact count.** It predicts `pos0` ~25% at 4 facts and ~6.25% at 16,
  which is a sharp falsifiable test and is the first half of r37. Until that runs, 11.4% vs 12.5% is a suggestive
  coincidence on a single point.
- **No fix has been tested.** The candidate follows from the diagnosis: with all 16 key letters drawn distinct, the
  key's *own* characters are unique, so `--value-shift 3` would make the slot keyed on `a` return `c` and the slot
  keyed on `b` return `d` — both addressable without ever querying `:`. It requires the model to route a read
  performed two positions earlier through attention, which is plausible and unproven.
- Tiled facts, 8 facts, one store size, char level. As r35.
