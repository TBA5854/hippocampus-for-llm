# r37 — is the ceiling really 1/n_facts, and can a shift address around it?

r36 found r35's 10% decomposes as `pos0` 11.4% x `pos1` 80.5%, with `pos0` sitting on **1/n_facts = 12.5%**, and
explained it: the first answer character must be retrieved by reading at the **`:`** token, which appears once per
fact and therefore matches every fact's slot equally.

That explanation rests on **one fact count**. 11.4% vs 12.5% is a suggestive coincidence until the curve is drawn.

## H37a — `pos0` tracks 1/n_facts

| n_facts | predicted `pos0` | predicted exact (`pos0` x ~0.83) |
|---|---|---|
| 4 | **25.0%** | ~21% |
| 8 (r36) | 12.5% -> measured **11.4 / 11.9%** | ~10% -> measured **10.3 / 9.1%** |
| 12 | **8.3%** | ~7% |

**Falsified if** `pos0` is flat across fact counts, which would mean 11.4% was a coincidence and the ambiguity story
is wrong. A partial slope (moving, but not as 1/n) would say ambiguity is one term among several.

The task caps at **13 facts**: keys and values are both 2 letters drawn from `randperm(26)`, so `2 x n_facts <= 26`.
4 / 8 / 12 is the widest clean sweep available, a 3x range.

## H37b — `--value-shift 3` addresses around the ambiguity

The fix follows from the diagnosis rather than from tuning. `make_batch` draws all key letters from one
permutation, so **every key character is unique across the fact set** — the ambiguity is specific to the delimiter.
For a fact laid out `a b : c d` at positions 0-4:

| shift | slot keyed on `a` returns | slot keyed on `b` returns |
|---|---|---|
| 1 (current) | `b` | `:` |
| **3** | **`c`** (answer char 1) | **`d`** (answer char 2) |

At shift 3 **both answer characters are addressable by unique key characters**, and `:` is never the retrieval
point. The query `?ab:` contains `a` and `b`, so both reads are performed — two and one positions before the
prediction point, and carried forward by causal attention within the query segment.

- **Predicted:** `pos0` breaks the 1/n_facts ceiling and rises toward `pos1`'s 80-86%.
- **Falsified if** `pos0` stays at ~12.5%, which would mean the model cannot route a read performed at an earlier
  position to a later prediction — a different and equally interesting limit, about *when* a read is usable rather
  than *what* it can address.
- **Also possible: `pos1` falls.** At shift 3 the slot keyed on `c` returns the character 3 later, not `d`. If total
  exact match does not improve, the shift has moved the problem rather than solved it, and that is the honest
  reading.

This is a **task-layout** fix, not an architecture fix, and it should be reported as one: it works by arranging for
every retrieval to key on a unique token. The underlying constraint from r36 stands either way — a position-free
single-token key can only address a single distinct token.

## Runs (box1, sequential, ~12 min each)

```
BASE = r36's LANG_BASE (= r35's = r23's)

r37_f4        --n-facts 4                    H37a
r37_f12       --n-facts 12                   H37a
r37_shift3    --n-facts 8 --value-shift 3    H37b, matched to r36_lang_s1 on every other flag
```

r36's 8-fact runs supply the middle point of the sweep and the shift-1 baseline, so no re-run is needed for either.

## What would make this round wrong

- Reading `pos0` off a run whose exact match is at chance — the split needs something to split. Checked first.
- Comparing `r37_shift3` against a differently-seeded baseline: it uses seed 1, matching `r36_lang_s1`.
- Treating a `pos0` rise at shift 3 as an architecture improvement. It is a layout change that avoids the delimiter;
  the addressing constraint is unchanged.
