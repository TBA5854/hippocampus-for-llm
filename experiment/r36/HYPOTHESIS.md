# r36 — where the second character goes, and what an empty-store read costs

Two independent questions, one per box, both named as next steps by the rounds that just closed.

## box1 — where does r35 lose the answer?

r35 got **10.0% exact / 47.1% char** from the store against a **97.2%** in-context ceiling. Those two numbers are
not consistent with independent character errors: if each character were right 47.1% of the time independently,
pairs would be right 22%, not 10%. So the failures are **correlated**, and the round could not say how.

The answer's two characters come from **different slots**, which is what makes this measurable. With
`value_shift 1`, a slot keyed on position `t` returns position `t+1`. For a fact written `ab:cd` and a query `?ab:`:

- the slot keyed on `:` returns **`c`** — the first answer character
- the slot keyed on `c` returns **`d`** — the second

Two retrievals, chained. At eval the model is teacher-forced, so `c` is present in the input when `d` is predicted:
the second hop is not blocked by the first being wrong, it is a separate lookup.

`answer_stats` now reports accuracy **per position within the answer** (one test). Three outcomes, and they mean
different things:

- **`pos0` high, `pos1` low** — the first hop works and the second does not. The store can find "what follows the
  colon" but not "what follows that", i.e. chained retrieval fails. That is a *mechanism* finding and it points at
  the read being usable once per answer.
- **both low together** — the match fails, not the content, and the 47.1% is carried by a subset of examples getting
  both. Then the r9-style "matching intact / content partial" story in r35's diagnosis is **wrong** and should be
  corrected.
- **`pos0` low, `pos1` high** — would mean the colon slot is the broken one specifically, which the tiling makes
  unlikely and would need its own explanation.

Also run: **seeds 1 and 2** of `r35_on`, because r35 is one seed and r34 showed a 3.3x magnitude spread across four
seeds on this very architecture.

- **H36a — `pos0` substantially exceeds `pos1`.** Chained retrieval is the limit. Falsified if they are equal.
- **H36b — 10.0% replicates within the spread r34 measured.** Falsified if a seed lands near chance, which would
  make r35's headline a lucky draw and is exactly the error this session already made twice on r34.

## box2 — does suppressing the empty-store read recover r34's lost net benefit?

r34's clearest actionable finding: at segment 0 the store is **empty**, the gate is open anyway (~0.165 on every
seed), and the model pays to read nothing — **-0.0081 nats on seed 1, -0.0024 on seed 2**, against +0.0006 and
+0.0018 on seeds 0 and 3. On seed 1 that cost cancelled the retrieval gain from all three later segments, which is
the whole reason its net benefit was +0.0006 while its retrieval gain was a healthy 2.2x the control.

`--suppress-empty-read` (new, default **off** so no earlier round changes) skips retrieval while `n_stored == 0`.

- **H36c — the segment-0 cost goes to zero and net benefit rises to meet the retrieval gain.** Predicted: `s0` gap
  ~0.000 by construction, later-segment gain unchanged (~+0.008), so net rises from +0.0058 mean toward +0.008.
  **Falsified if** the later-segment gain also drops — which would mean the empty read was doing something useful
  after all, e.g. acting as a learned bias the model relies on.
- **H36d — it helps most on the seeds that paid most.** Seeds 1 and 2 carried the cost; seeds 0 and 3 did not.
  Running seeds 1 and 0 tests the interaction: a large gain on seed 1 and roughly nothing on seed 0.

Matched against r34's existing `r34_off_s1` / `r34_off_s0` controls — same seeds, same steps, same everything but
the new flag, so the comparison is exact and no new control is needed.

## What would make this round wrong

- Comparing the box2 runs against a control at a different seed. Each is scored against `r34_off_s{N}`, its own.
- Quoting box1's `pos0`/`pos1` split from a run whose exact match is near chance: the split is only meaningful
  where there is something to split. Checked before it is read.
- The per-position metric assumes the answer mask is contiguous and ordered, which it is for this task (the answer
  ends the segment). It would not be meaningful for a task with scattered answer positions.
