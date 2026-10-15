# r39 — store size is not the lever; deduping the store HURTS on language (unlike MQAR)

## Q1 — store size vs the fact-cliff

| run | store M | exact | char | pos0 |
|---|---|---|---|---|
| `r36_lang_s1` — 8 facts | 512 | 10.3% | 45.9% | 11.4% |
| `r39_m1024_f8` — 8 facts | 1024 | **10.3%** | 45.9% | 11.4% |
| `r39_m2048_f8` — 8 facts | 2048 | 11.7% | 51.3% | 12.2% |

**H39a — flat. Confirmed, emphatically.** M=1024 is *byte-identical* to M=512 (10.3%, char 45.9%, pos0 11.4% — the
extra 512 empty slots change literally nothing), and M=2048 moves it by 1.4 points, within noise. Doubling or
quadrupling the store does not touch the fact-cliff. The 8 facts already fit at M=512 (~381 slots written into 512);
the limit is learning to disambiguate them (r36/r38), not room. This is r13's MQAR result — store size delays
lift-off, never caps recall — confirmed on the language task.

## Q2 — is a deduped store's read near zero?

| run | slots kept | exact | char | pos0 |
|---|---|---|---|---|
| `r37_f4` — 4 facts, full | 100% | **90.3%** | 95.1% | 94.1% |
| `r39_f4_frac25` — surprisal 25% | 25% | **20.9%** | 56.1% | 25.8% |
| `r39_f4_frac10` — surprisal 10% | 10% | 15.8% | 44.1% | 24.7% |

**Not near zero — but dedup HURTS here, which is the opposite of MQAR.** 20.9% is still ~140x the 0.15% exact chance,
so the read does not collapse. But it falls hard from 90.3%, where on MQAR the same 25% surprisal gate was *free*
(r30: 100% vs 99.7%). The tell is `pos0`: it drops from 94.1% to **25.8% = 1/n_facts**. Deduping removes the very
redundancy that let the model disambiguate the shared `:` key — with only 25% of slots kept, the colon-slots thin
out and the read falls back to a uniform draw over the 4 facts. So on language the tiling redundancy is
*load-bearing* for disambiguation, precisely because the keys are ambiguous (r36); on MQAR it was not, because the
keys were already distinct.

## Diagnosis

Both answers point at the same root as r36/r37: the language bottleneck is **key ambiguity**, not capacity. More
slots do not help (nothing is full); fewer slots hurt (they remove the redundancy that fights the ambiguity). The
lever is disambiguation — shift-3 (r37) or the curriculum (r38/r40) — not store geometry.

## What this does NOT show

- One seed each. The store-size flatness is robust (M1024 identical to M512); the dedup numbers are single runs.
- Dedup interacts with the fact-cliff: this is 4 facts (the solvable case). Whether a *curriculum* + dedup keeps the
  slots that matter is untested.
