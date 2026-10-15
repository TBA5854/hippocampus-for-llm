# r34 — PG-19: does the block help on natural text, and can the claim survive the "parameters alone" trap

## Ideate

Every round from r1 to r33 measured recall on a synthetic task (MQAR, or persona facts). That was the right
instrument — it isolates binding and retrieval, and it is where the failure modes were found. But it also means the
whole project's evidence is from a task built to need memory. The user's queue put the natural-language case next:

> "yeah after current, this, then episode, then go for pg 19, thats the long convos if i remember"

PG-19 is the DeepMind long-range LM benchmark: Project Gutenberg books published before 1919, ~69k words per book,
built precisely because book-length context defeats a fixed window. It is the honest test of the design's claim,
because nothing in a novel is planted for the memory to find.

## The trap this round has to avoid

The report already flags it: a memory-augmented model has more parameters than a memory-disabled one, so **a better
perplexity proves nothing by itself** — the read head, write head and their projections could simply be extra capacity
spent on local prediction, and a memory block that never retrieved anything would still win the comparison.

Two controls answer it, and the second is the load-bearing one:

1. **Matched disabled control** (`--hc off`), same everything else. Necessary, not sufficient, for the reason above.
2. **The per-segment profile — a within-model control.** `model.forward(idx)` carries no KV cache across segments
   ([`mech/model.py:80`](../../mech/model.py#L80)); the store is the *only* channel between one segment and the next.
   So for a memory-off model, segment index is meaningless: segments 1..n-1 are statistically identical draws to
   segment 0 and the loss profile must be flat. For a memory-on model, segment 0 is read against an **empty store** and
   later segments are read against a filled one.

   **A downward slope in loss across segment index cannot be bought with parameters.** It is the same weights, the same
   token distribution, the same position-in-segment — the only thing that changed is how much is in the store. That
   makes the slope, not the absolute perplexity, the round's actual measurement.

`run_example` already returns `per_seg` and `evaluate` already averages it, so the primary metric costs no new code.

## Built this round

- `BookCorpus` + `load_books` in [`mech/data.py`](../../mech/data.py): PG-19 as one tensor **per book**, sampled in
  proportion to length. An example never straddles two books — a flat concatenation would ask the store to carry
  context across a boundary where none exists, and the resulting null would look like "memory does not help".
  Vocabulary is built from the training books only; the fraction of held-out characters outside it is printed rather
  than assumed negligible. Seven tests, including the straddle test and the length-proportional test.
- `--data-books` / `--max-books` in [`mech/train.py`](../../mech/train.py).
- Corpus: PG-19's own **test** split (100 books, 40 MB) as the training corpus and its **validation** split
  (50 books, 18 MB) as held-out. The real train split is 28,602 books / ~11 GB, far past this budget. No book is in
  both. The directories keep PG-19's split names so the substitution stays visible.

## Hypotheses

- **H34a — the memory-on model's per-segment loss slopes down and the memory-off model's is flat.** This is the claim.
  Predicted: segment 0 roughly equal between the two models (empty store either way), and the gap opening with segment
  index. **Falsified if** the memory-on profile is flat too — which would say the store is written but never usefully
  read on natural text, i.e. the whole synthetic result does not transfer.
- **H34b — a scrambled store destroys the slope but not the parameter count.** `--shuffle-store` at matched settings
  keeps every weight and every FLOP and randomises what is retrieved. **Falsified if** the slope survives scrambling,
  which would mean the "memory" benefit is the read head acting as extra local computation, and H34a's slope would
  then be an artefact rather than retrieval.
- **H34c — validation perplexity improves, by less than the slope suggests.** Segment 0 gets nothing, so the whole-run
  average dilutes the effect by roughly 1/n_seg. Reported, but **not** the headline, precisely because it is the
  number the parameter-count objection applies to.

A null on H34a is a real outcome and gets written up as one: it would mean that exact-token retrieval, which is what
this store does well (r31d: the key *is* the token embedding), is not what natural-text prediction is short of.

### Pre-registered: how a null gets diagnosed, decided BEFORE the result

There are two different nulls and they must not be conflated, so the rule is fixed in advance:

- **Gate open, profile flat** -> the honest null. The store is written, read, and contributes nothing on natural text.
  That bounds the whole project to tasks with plantable bindings, and gets written up as such.
- **Gate shut (`gates=1:0.000`) and profile flat** -> **not** evidence about natural text at all. It is the
  bootstrapping deadlock this project already named and already solved: r19-r25 found store size, fact count, key
  vocabulary and write rate *all* fail cold for this one reason, and a curriculum escapes all four. At step 60 and
  step 250 of `r34_mem` the gate is `0.000`, so this branch is live and is the one to expect.

The escape, if the second branch happens, is r35 and it is the known one applied to this corpus: warm-start from a
setting where the store is *demonstrably* worth reading, then transfer to plain PG-19. The natural easy rung on book
text is a **repeat** curriculum — segment 3 reproduces segment 0's passage, so retrieval has obvious value and the
gate has a gradient to open on — followed by a transfer run on unmodified PG-19 with the gate already open. That
tests the right thing: whether a gate opened on real text *stays* open and keeps paying once the repeats are removed.

Writing this down before the runs finish is the point. "The gate never opened" is an explanation that would be very
easy to reach for after seeing a flat line, and pre-registering it as a *diagnosis with its own next experiment*
keeps it from becoming an excuse.

### Correction from the running logs: compare profiles, do not assume flatness

The argument above says a memory-off profile "must be flat". The control's own logs say otherwise, mildly but
**systematically**: segment 2 is the highest in every eval of `r34_off` (step 1000 `[1.875 1.880 1.895 1.876]`,
step 2000 `[1.525 1.527 1.549 1.533]`), and the same bump appears in `r34_mem`. With no KV cache the model cannot
know which segment it is in, so this is a property of the sampler or the corpus, not of memory.

It is small (~0.02 nats) but it is not noise, and it would be easy to read a matching bump in the memory-on run as
signal. So the metric is tightened: **the measurement is the difference between the two profiles**, segment by
segment, not the memory-on profile against an assumed flat line. Any artifact common to both cancels. Flatness is
the control's job to demonstrate, not the analysis's to assume.

Also noted: `gates=1:0.027` at step 1000 against `0.000` at steps 60 and 250 — the gate IS opening, so the deadlock
branch may not be the one that fires. Recorded here because the prediction above expected it to.

## The runs

Four, matched on seed, steps, optimiser, segmentation and data:

```
common: --task lm --data-books data/pg19 --n-seg 4 --seg-len 256 --steps 8000 --seed 0
        --d-model 256 --n-layer 6 --batch 24

  1  --hc off                                            control: no memory
  2  --hc on --read-layers 1 --write-layer 5 --memory-size 1024 --beta-grad 4 --dg-bypass
  3  --hc on ... --shuffle-store                         control: retrieval scrambled, params identical
  4  --hc on ... --write-select surprisal --write-frac 0.25 --memory-size 256
                                                         r30's compression, on natural text
```

Run 4 asks the second question of the round: the 4x write-rate compression was established on a task where the
valuable positions are a handful of planted facts. On a novel there are no planted facts, so a surprisal gate has to
find its own. If run 4 holds run 2's slope at a quarter of the store, compression transfers; if it collapses, the
compression result is task-specific and that bounds it.

## What would make this round wrong

- Reading the slope off a run whose store never filled (then later segments differ only in noise) — so `n_stored` and
  the gate magnitude are logged per segment and checked before any slope is quoted.

  **Checked, with the real config rather than arithmetic:** slots after each segment are
  **255 -> 510 -> 765 -> 1020** of 1024. The store fills monotonically and never overflows, so segment 0 reads an
  empty store, each later segment reads strictly more than the last, and no eviction runs. That is the precondition
  the slope argument needs, and it also says the effect should *grow* with segment index rather than appear as a step.
- Quoting perplexity against a control at a different `beta_grad` / `dg_bypass` / seed. Matched, as every round since
  r1.
- The held-out books being unreachable from the train vocabulary. Printed at startup.
