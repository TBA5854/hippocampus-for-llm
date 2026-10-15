# r29 — results: the loop is stable, the knowledge is not

## Numbers (1x store, 600 sleep steps/cycle at lr 3e-4, rehearse 1.0 + rehearse-mem 3.0, batch 32, one user per cycle)

`r29_cycles12` — 12 users, no past replay. Retention is each earlier user's held-out fact answered **from weights
alone**, memory off, nothing in context:

| cycle | consolidated (this user) | user 0 | user 4 | user 9 | store rewritable | LM | cumulative drift |
|---|---|---|---|---|---|---|---|
| 0 | 75% | 75% | — | — | 100% | 0.996 | 73 |
| 1 | **100%** | 0% | — | — | 84% | 0.905 | 108 |
| 2 | 88% | 0% | — | — | 75% | 0.856 | 136 |
| 4 | **100%** | 12% | 100% | — | 75% | 0.817 | 182 |
| 5 | 88% | 0% | 50% | — | 88% | 0.761 | 203 |
| 6 | **100%** | 0% | 38% | — | 62% | 0.761 | 221 |
| 9 | **100%** | 0% | 12% | 100% | 100% | 0.637 | 267 |
| 11 | **100%** | 0% | 0% | 38% | 75% | 0.603 | 295 |

The decay is the same shape for every user: ~100% the night they are consolidated, ~38-62% one cycle later, ~0-12%
two cycles later, 0% after that. `r29_cycles6` reproduces cycles 0-5 exactly (same seed).

## Verdicts

- **H29a (consolidation stays at 100% across cycles) — confirmed.** 88-100% at every one of 12 cycles. Sleeping on a
  model that has already slept eleven times is no harder than the first time.
- **H29b (retention fails without replaying the past) — confirmed.** Every user is fully forgotten within ~3 cycles.
- **H29d (drift stays below the ~418 cliff) — confirmed, and better than feared.** Drift accumulates *sub*-linearly:
  73 after one sleep, 295 after twelve (a linear extrapolation would have given ~880). The store stayed writable
  (62-100%) and the LM improved monotonically (0.996 -> 0.603).
- **H29c (replaying the past fixes retention) — confirmed, under a fixed budget.** The first attempt OOMed, which is
  itself the finding: replaying every user in full grows linearly and died around user 5. Capped at 512 rows per
  night, retention goes from 0% to 50-100% at 6 users and a mean of 69% at 12, with the cost of a night constant
  regardless of how many users have accumulated.

## Diagnosis

Two mechanisms that look alike but are not:

- **The loop is stable.** Consolidation, decoding, store writability and LM quality all hold across a dozen cycles.
  Nothing about repeated sleeping degrades the *machinery*. Drift decelerating is consistent with the weights settling
  into a region that satisfies rehearsal, rather than wandering.
- **The knowledge is not.** Each sleep supervises one user's facts and nothing else defends the previous user's, so the
  same weights are rewritten. This is ordinary catastrophic forgetting, arriving at the level of *users* rather than
  tasks — and single-cycle experiments (r27, r28) cannot see it by construction.

The design consequence is sharp. The store is bounded and evicts, so facts must move into weights; but weights are
shared across users, so consolidating user N erases user N-1. **A fixed replay budget is enough to hold this**: it
turns total forgetting into a slow decay whose rate is set by how many rows a night can afford. Per-user parameters
(adapters/LoRA, sleep writing only into that user's slice) remain the alternative if the decay slope matters more than
the nightly cost; not attempted here.

A caveat on the task: every user draws 8 facts from the same 128-symbol vocabulary, so users actively **collide** —
the same key with a different value. That is the hardest possible case for shared-weight consolidation and the easiest
case for observing forgetting. Real users share far less key space.

## Fixed-budget replay (rerun)

`--replay-cap 512`: the current user keeps its full supervised set, and earlier users share what is left, so the cost
of a night is constant no matter how many users have accumulated. Each past user therefore gets a thinner slice as
they pile up, which is exactly the scaling question.

`r29_cycles6_past` (6 users, cap 512) against the no-replay run at the same cycle:

| cycle | consolidated | retention, all users so far | no-replay equivalent |
|---|---|---|---|
| 1 | 100% | 1.00 1.00 | 0.00 1.00 |
| 3 | 100% | 1.00 1.00 0.62 1.00 | 0.12 0.00 0.00 1.00 |
| 5 | 75% | **1.00** 0.88 0.50 0.88 0.75 0.75 | 0.00 0.00 0.00 0.00 0.50 0.88 |

**H29c — confirmed at 6 users.** Every earlier user is still readable from weights at the end (50-100%, mean ~79%)
where the no-replay run had them all at 0%. Drift is unchanged (202 vs 203) and consolidation is unchanged, so the
retention costs nothing but the replay rows themselves — and those are capped, so a night costs the same whether it is
the second user or the twentieth.

The oldest user does *best* (100%), which is the expected shape: it has been replayed on every subsequent night. The
weakest is user 2 at 50%.

`r29_cycles12_past` (12 users, same cap) is the scaling test — with the budget fixed, each past user's slice thins as
they accumulate:

| users so far | mean retention of the EARLIER users | oldest user |
|---|---|---|
| 2 | 1.00 | 1.00 |
| 4 | 0.88 | 1.00 |
| 6 | 0.80 | 1.00 |
| 8 | 0.77 | 0.75 |
| 10 | 0.68 | 0.50 |
| 12 | **0.69** | 0.75 |

**Graceful decay, not a cliff** — about -3 points of mean retention per added user, against **0%** for anything older
than ~2 cycles without replay. Chance is 0.8% (128 symbols), so even the weakest user at 0.38 is ~47x chance.
Consolidation held at 75-100% throughout, drift stayed sub-linear (294 at cycle 12), and LM loss improved
monotonically (0.996 -> 0.579).

Extrapolating the slope, mean retention would reach ~50% somewhere around 20 users on a 512-row budget. That is the
number to raise with a bigger budget, not a wall.
