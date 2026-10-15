# r38 — control the 4-fact result, then try the curriculum on it

r37 found a **cliff between 4 and 8 facts** in the text vocabulary: 4 facts 90.3% exact with `pos0` 94.1%, 8 facts
10.3% with `pos0` 11.4% sitting on 1/n_facts. The 1/n line is the signature of a model that has not learned to
disambiguate the shared `:` key and defaults to a uniform draw — not a combinatorial wall, because at 4 facts the
same architecture disambiguates almost perfectly.

## H38a — the 4-fact result survives its control

**90.3% is currently uncontrolled.** `r35_shuf` was run at 8 facts and r37 added no control of its own. A matched
`--shuffle-store` at 4 facts must sit at chance (exact 0.15%, char 3.8%, CE 3.258), exactly as it did at 8 facts
where it landed on 0.2% / 3.8% / 3.259.

**Falsified if** the 4-fact scrambled control clears chance — which would mean 4 facts is partly solvable without
the store (four `k:v` pairs tiled through a segment is a small space, and some other route may exist), and 90.3%
would need reinterpreting. This runs **first** because everything else in the round depends on it.

## H38b — a curriculum carries the 4-fact solution to 8 facts

r25's result applied to the language task. r25 established the fact-count cliff is **not a capacity limit**: 16
facts go 3.8% cold -> 100% warm-started, 48 facts 0.9% -> 100%. The mechanism is a gate/decoder bootstrapping
deadlock, and warm-starting from a solved easier setting escapes it.

r37's gate magnitudes are consistent with that reading — 0.275 at 4 facts, 0.206 at 8, 0.147 at 12: the gate opens
less the further past the cliff the run begins.

Warm-start 8 facts from `r37_f4`'s checkpoint via `--init`.

- **Predicted:** `pos0` breaks the 12.5% line and exact match goes well above r36's 10.3%.
- **Falsified if** it lands at ~10% like the cold run. That would separate this cliff from r24/r25's — same
  signature, different cause — and say the delimiter ambiguity is a real wall in the text vocabulary rather than a
  bootstrapping failure. **That is a publishable negative**, because it would mean the project's central escape
  route does not generalise off MQAR.

## H38c — the shift and the curriculum compose

`--value-shift 3` reached 31.6% cold at 8 facts (r37). If both interventions address the same bottleneck by
different means, warm-starting a shift-3 run from a 4-fact shift-3 model should beat both; if they address different
bottlenecks the gains should be roughly independent. `r38_f4_s3` supplies that warm start and also puts a **second
fact count on the shift-3 axis**, so the shift result stops resting on one point.

## Runs (box1, sequential, ~12 min each)

```
BASE = r37's (= r36's = r35's = r23's), seed 1

r38_f4_shuf   --n-facts 4 --value-shift 1 --shuffle-store         H38a, the missing control
r38_curr8     --n-facts 8 --value-shift 1 --init r37_f4.pt        H38b, the curriculum
r38_f4_s3     --n-facts 4 --value-shift 3                         H38c prep + 2nd shift-3 point
```

## What would make this round wrong

- Running the curriculum before its control. The order above is deliberate.
- Reading a warm-start gain as "the curriculum works" if the 4-fact control fails — then what is being carried
  forward is not a solution.
- `--init` loads a checkpoint trained at a different `n_facts`. Store tensors are non-persistent buffers so this is
  the same mechanism r19/r25 used, and the vocabulary matches (same PG-19 books, same `TASK_CHARS`).
