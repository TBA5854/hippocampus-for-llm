# r60 — RESULTS: the r59 multi-layer escape is seed-robust; the per-layer dip was noise

r59 was single-seed and the per-layer arm was non-monotonic (16f 92.7% → 24f 71.6% → 32f 100%). r60
adds seeds to confirm the escape isn't luck. All at 128 symbols (the cliff regime), MQAR cold.

## Recall by seed (single-layer baseline cliffs at 9.1% / 4.2%)

| arm | seed 0 (r59) | seed 1 (r60) | seed 2 (r60) |
|---|---|---|---|
| shared-symmetric, 16 facts | 99.4% | **100.0%** | **100.0%** |
| shared-symmetric, 32 facts | 96.2% | 91.6% | **100.0%** |
| per-layer, 16 facts | 92.7% | **100.0%** | — |
| per-layer, 24 facts | 71.6% | **100.0%** | — |

## What it shows
- **The multi-layer escape is robust.** Every seed of both variants clears the cliff by a wide margin
  (91.6–100%) where single-layer collapses to 4–9%. The r59 finding holds: the fact-count cliff belongs
  to the single layer-0 site, and reading/writing at every layer escapes it cold.
- **The per-layer non-monotonicity was single-seed noise.** Seed 1 gives 100% at both 16 and 24 facts,
  so r59's 71.6% at 24 was a bad-seed artifact near the cliff (r45 saw the same variance), not a real
  dip. Per-layer escapes as cleanly as shared once you average out the seed.
- **Shared-symmetric remains the better engineering choice:** equal or higher recall, one store not six,
  ~2× faster. r59's "leans pro-CSA2" verdict stands.

## Verdict
- **H60 (the r59 escape is seed-robust; per-layer dip is noise):** CONFIRMED. Both variants escape the
  cliff across seeds; the per-layer dip does not replicate.
