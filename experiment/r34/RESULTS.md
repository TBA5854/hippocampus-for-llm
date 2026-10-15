# r34 — results: a small retrieval effect on PG-19, real across 4 seeds, magnitude varying 3x

> **Verdict after four seeds.** Every seed's later-segment gain exceeds the scrambled control, by 2.2x at worst and
> 7.3x at best. The effect is real and it is retrieval. It is also **small and variable**, and one seed's *net*
> benefit is negligible.
>
> | seed | later-segment gain | segment-0 change | net |
> |---|---|---|---|
> | 0 | +0.0102 | +0.0006 | +0.0078 |
> | 1 | **+0.0035** | **-0.0081** | **+0.0006** |
> | 2 | +0.0082 | -0.0024 | +0.0056 |
> | 3 | +0.0116 | +0.0018 | +0.0091 |
> | **mean** | **+0.0084** | -0.0020 | **+0.0058** |
> | `r34_shuffle` (control) | +0.0016 | +0.0014 | +0.0016 |
>
> **This file was wrong twice before it was right, and both are recorded below rather than tidied away.** It first
> reported seed 0 alone as the result. It then over-corrected to "does not replicate" on the strength of seed 1
> alone — the same single-seed error in the other direction. Four seeds say: real, ~5x the control on average,
> 3.3x spread in magnitude (+0.0035 to +0.0116), and a net benefit that seed 1 shows can be eaten entirely by the
> segment-0 cost.
>
> **Every control is still seed 0 only** (`r34_shuffle`, `r34_compress`), so the 48x scramble collapse and the
> compression result each rest on one seed of a pair.

PG-19, char level. 100 books / 40.4M tokens training, 50 held-out books / 17.4M. 4 segments x 256 tokens, store
carried across segments within one book, 10,000 steps, seed 0. Chance is not a meaningful reference for an LM task,
so every number is against a **matched control**.

## The numbers

| run | params | val loss | val ppl | profile s0..s3 | **effect** | gate |
|---|---|---|---|---|---|---|
| `r34_off` — no memory | 4.86M | 1.3047 | 3.687 | 1.296 1.305 1.316 1.305 | — | — |
| **`r34_mem`** — memory on | 5.13M | **1.2967** | **3.657** | 1.296 1.296 1.305 1.294 | **+0.0096** | 0.164 |
| **`r34_compress`** — 25% write rate, 1/4 store | 5.13M | **1.2975** | **3.660** | 1.297 1.296 1.305 1.295 | **+0.0103** | 0.126 |
| `r34_shuffle` — store scrambled | 5.13M | 1.3031 | 3.681 | 1.295 1.303 1.314 1.304 | **+0.0002** | 0.001 |
| `r34_mem_misconfigured` — query/key in different spaces | 5.13M | 1.3035 | 3.682 | 1.297 1.303 1.313 1.303 | +0.0033 | 0.191 |

**effect** = the round's pre-registered statistic: how much *more* the later segments improve over segment 0 than
they do in the memory-free control. Eval jitter over the last five evals is 0.003.

### The statistic is ambiguous, and seed 1 is what exposed it

`effect = slope(memory) - slope(control)` with `slope = s0 - mean(s1..s3)`. That number rises for **two different
reasons**: the later segments improving, or **segment 0 getting worse**. Those are not the same claim, and only the
first is retrieval. Splitting it:

| run | later-segment gain | segment-0 change | **net** (what a user gets) |
|---|---|---|---|
| `r34_mem` (seed 0) | **+0.0102** | +0.0006 | **+0.0078** |
| `r34_compress` (seed 0) | +0.0096 | -0.0007 | +0.0071 |
| **`r34_mem_s1` (seed 1)** | **+0.0035** | **-0.0081** | **+0.0006** |
| `r34_shuffle` (control) | +0.0016 | +0.0014 | +0.0016 |
| `r34_mem_misconfigured` | +0.0022 | -0.0011 | +0.0013 |

Seed 1's headline effect (+0.0115) is *larger* than seed 0's (+0.0096) while its actual benefit is **a third of it**,
because the number is carried by segment 0 degrading by 0.008 rather than by later segments improving. Seed 1's
later-segment gain, +0.0035, sits between the scrambled control (+0.0016) and seed 0 (+0.0102) — it is not cleanly
separated from a store whose contents are random.

**The segment-0 cost is itself a finding, and it is what makes the net benefit variable.** At segment 0 the store is
empty, the gate is open anyway (~0.165 in every seed), and the model pays for reading it: **-0.0081, -0.0024,
+0.0006, +0.0018** across seeds 1, 2, 0, 3. On seed 1 that cost cancels the benefit of all three later segments,
which is the whole reason seed 1's net is +0.0006 while its retrieval gain is a healthy 2.2x the control.

So the later-segment gain and the net benefit are two different claims, and only the first replicates cleanly:

| | mean | min | max | control |
|---|---|---|---|---|
| later-segment gain (retrieval) | +0.0084 | +0.0035 | +0.0116 | +0.0016 |
| net benefit (what a user gets) | +0.0058 | +0.0006 | +0.0091 | +0.0016 |

**An open gate on an empty store is not free, and nothing in the design prevents it.** The cheap fix is to suppress
the read when `n_stored == 0`, which should convert the retrieval gain into net benefit on every seed. Untested, and
the obvious follow-up.

Per-segment gap against the control (positive = memory arm is better at that segment):

| run | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| `r34_mem` | **+0.001** | +0.009 | +0.011 | +0.011 |
| `r34_compress` | **-0.001** | +0.009 | +0.011 | +0.010 |
| `r34_shuffle` | +0.001 | +0.002 | +0.002 | +0.001 |

## Verdicts

- **H34a (the memory profile slopes down relative to the control) — supported on 4/4 seeds, with 3.3x spread.**
  Later-segment gain +0.0102 / +0.0035 / +0.0082 / +0.0116 against the scrambled control's +0.0016. Net benefit is
  where the seeds disagree: +0.0078 / +0.0006 / +0.0056 / +0.0091, because the segment-0 cost varies. Seed 0's
  reading was +0.0096, about 3x the eval jitter,
  and with the required shape: **segment 0 is at +0.001, i.e. nothing**. That is forced — segment 0 reads an empty
  store (measured fill: 255 -> 510 -> 765 -> 1020 of 1024 slots) — and it is the part a parameter-count objection
  cannot touch. The benefit appears at segment 1, exactly when the store first has contents.
- **H34b (scrambling destroys it) — confirmed, and this is the load-bearing control.** `r34_shuffle` has the **same
  5.13M parameters, the same FLOPs and the same read path**; only the retrieved content is randomised. Effect falls
  from +0.0096 to **+0.0002**, a 48x reduction, and the model **closes the gate to 0.001** — it learns the store is
  not worth reading. Capacity therefore does not explain the memory arm's result.
- **H34c (perplexity improves by less than the slope suggests) — confirmed.** 3.687 -> 3.657, about **0.8%**. Segment
  0 gets nothing and dilutes the average, which is exactly why perplexity was not the headline.

## The compression result transfers to natural text

`r34_compress` writes **25% of positions into a 256-slot store** against `r34_mem`'s 1024, and its effect is
**+0.0103 against +0.0096** — if anything slightly better, certainly not worse. r30 established 4x write-rate
compression on a synthetic task built around planted facts; the obvious worry was that a surprisal gate has nothing
to find in a novel. It does. **4x compression is free here too**, in a domain with no planted facts at all.

That also sharpens r31d's negative: slot *width* compression fails (chance), slot *count* compression is free. The
two axes remain non-interchangeable, now on two very different tasks.

## Diagnosis — what the effect is, and what it is not

The gap **plateaus**: +0.009, +0.011, +0.011 while available slots grow 255 -> 510 -> 765. If the model were
retrieving more as more became available, the gap should keep widening. It does not. So the benefit is **from having
a store at all, not from having a large one** — consistent with `r34_compress` matching `r34_mem` at a quarter the
slots, and with the plateau being the honest reason *why* compression costs nothing here.

**Gate magnitude is not evidence of usefulness.** `r34_mem_misconfigured` opens its gate **widest of all (0.191)**
and delivers +0.0033 with a flat profile, while `r34_mem` at 0.164 delivers +0.0096. A model will open a gate onto a
store it cannot use. This weakens the gate-open/gate-shut branch this round pre-registered for diagnosing a null:
a shut gate is informative (`r34_shuffle` at 0.001), an open one is not.

## The round's own error, recorded because it nearly became the result

The first memory runs used `--read-layers 1 --write-layer 5` with none of `--key-layer 0 --key-pos-free
--query-init identity --freeze-query`. That makes the key a last-block state and the query a random learned
projection of a layer-1 state — two different spaces. Every working configuration since r9 shares one space
deliberately: a position-free layer-0 key **is** the token embedding, and a frozen identity query matches it
near-exactly (r31d: destroy that exactness and recall goes to chance).

It produced effect **+0.0033 with a flat profile** — a clean null that was about to be written up as "the memory
block does not help on natural text". It was measuring the configuration, not the corpus. The placement had been
taken from the plan's prose ("read at an early layer, write at the last block"), which is true but omits the four
flags that make it work.

The run is kept as `r34_mem_misconfigured` rather than deleted: a same-corpus, same-budget, same-parameter negative
for a non-shared-space placement is a real data point, and it belongs to r35's question.

> **Rule this earns: take the placement from a working `run.sh` (r23, r30), never from prose.**

### And the corrected config was still not the working one

Found while building r35 by diffing against [`r23/run.sh`](../r23/run.sh) properly. Even after the placement fix,
every r34 memory run was missing:

- **`--value-skip-layer 0 --value-skip-norm`** — the normalised block-0 skip. r17/r18 established the last-block
  write does not work without it: plain skip is 100% at 1x and **28.1% at 2x**, normalised is **99.4% at 2x**. r34
  had no skip path at all, so the value written at the last block is the one r9 measured at **2.3%** on MQAR.
- `--readout graded`, `--beta 32 --beta-max 64 --n-steps 0`, `--write-threshold 0.6`.

So r34's +0.0084 is a **lower bound measured through a handicapped write path**, not the architecture's answer on
PG-19. That does not invalidate the round — the controls are matched, so the comparison is sound and the effect is
real — but the magnitude should not be quoted as what this design can do on natural text. Re-running the PG-19 LM
arm on the r23 base is a cheap and clearly-motivated follow-up.

## What this does NOT show

- **Four seeds of the memory arm, one seed of every control.** The retrieval gain replicates 4/4; the *net* benefit
  ranges from +0.0006 to +0.0091 because the segment-0 cost varies, so "0.8% better perplexity" is the best seed,
  not the expected one. The mean is +0.0058, i.e. ~0.4%.
- **The write path was handicapped** (no normalised block-0 skip — see above), so every number here is a lower
  bound.
- **Every control is seed 0 only.** `r34_shuffle` and `r34_compress` have no replication at all, so the 48x
  scrambled-store collapse and the "compression is free" result each rest on a single seed of a pair.
- **The effect is small in absolute terms** — 0.8% perplexity. It is structured and it survives its controls, but it
  is not a capability result, and nothing here shows a fact being recalled from a book.
- **No rare-token breakdown.** The plan asked for one; the per-segment profile was the stronger within-model control
  and was done first. Which tokens the gap comes from is unmeasured, and is the natural r36: if the benefit is
  retrieval, it should concentrate on rare and repeated strings (names, places) rather than spread evenly.
- **Char level, 5M parameters, 4x256 tokens.** PG-19 is a benchmark for models with far longer contexts; this is the
  architecture's question asked on PG-19's text, not a benchmark result.
- The plateau is described, not explained. Whether a larger store would help with a longer horizon, or whether the
  read saturates, is untested.
