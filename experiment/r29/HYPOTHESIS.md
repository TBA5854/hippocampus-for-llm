# r29 — the eviction loop in steady state

## Ideate (from r27, r28)

One wake -> sleep works: consolidation 0% -> 100%, all slots reclaimable, LM improved, and at the 8x store too. Every
number so far is a **single cycle**, which cannot see the failure modes a memory that runs all day would actually hit:

- **Retention.** Sleep n+1 moves the same weights that hold user n's facts. Nothing so far has checked whether the
  first user's facts survive the second user's sleep. This is the question single-cycle runs structurally cannot
  answer, and the one most likely to break.
- **Drift accumulation.** r27 found a cliff — at weight drift ~418 the store becomes unreadable. A single sleep moves
  ~100-130. Six sleeps could cross it even if each is individually safe.
- **Throughput.** Slots reclaimed per cycle vs slots filled per wake. Consolidation at 100% frees everything, so the
  loop is stable *if* consolidation stays at 100% as cycles accumulate.
- **Livelock.** Facts that never consolidate pin slots. Cannot appear until consolidation drops below 100%.

`mech/sleep_cycles.py` (written this round) runs N cycles, one user each: fresh facts -> wake -> decode -> sleep ->
verify **every user so far** with memory off and nothing in context.

## Hypotheses

- **H29a — consolidation stays at 100% across cycles.** Falsified if it decays: sleeping on a model that has already
  slept is harder than sleeping on a freshly trained one, which would make the loop self-limiting.
- **H29b — retention fails without explicit replay of earlier users.** The strong prior from the forgetting literature
  and from r27's own LM result (unrehearsed sleep nearly tripled LM loss). Predicted: user 1's facts decay sharply
  after two or three later sleeps. Falsified if retention holds at >= 90% with no past-replay at all — which would be
  the more interesting result, and would say the facts land in weights that later sleeps do not disturb.
- **H29c — replaying earlier users' transcripts fixes retention** (`--replay-past`), at a cost in sleep budget that
  grows linearly with the number of users. This is where the design's real scaling limit should appear: a day's worth
  of users cannot all be replayed every night.
- **H29d — cumulative drift stays below the ~418 cliff** across 6 cycles, so the store remains writable and readable
  throughout.

## Falsifiers worth naming

- Consolidation holding at 100% while retention collapses -> the memory is a one-user-at-a-time device, and the
  eviction design needs per-user parameter isolation (adapters/LoRA), not a shared sleep.
- Drift crossing ~418 -> the loop has a hard cycle budget, after which the model must be reset or the store rebuilt.

## Runs

| box | name | change |
|---|---|---|
| box2 | r29_cycles6 | 6 cycles, no past replay — does retention fail at all? |
| box2 | r29_cycles6_past | 6 cycles, `--replay-past 1.0` — does replay fix it? |
| box1 | r29_cycles12 | 12 cycles, no past replay — where does drift end up, and does consolidation decay? |

All at the 1x store (fast enough for 12 cycles), 600 sleep steps (r27 reached 100% consolidation by 400), lr 3e-4, rehearse 1.0, rehearse-mem 3.0 (r28's best
configuration), batch 32, one user per cycle.
