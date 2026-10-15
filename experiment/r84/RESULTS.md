# r84 — CoT-over-memory WORKS. Route A + `write_skip_pad`: 2-hop 96%, broken control at the floor, seed-replicated, holds to 3-hop.

**The north star lands.** A model reasons across facts held **out-of-window** in the
bounded single store by emitting each intermediate as a token and re-querying — every
hop a clean single-hop store lookup. `write_skip_pad` removed the r83 intermediate
key-and-value collision, lifting hop2 from 45% to 98%. Broken-chain control at chance
proves it is genuine composition, not a leak; it replicates across seeds and extends to
3 hops.

## Numbers (final @8k, teacher-forced; per-hop = char position accuracy)
| run | hops | seed | recall (joint) | best | per-hop char |
|-----|------|------|----------------|------|--------------|
| r84_chain        | 2 | 0 | **96.4%** | 96.6% | pos0 98.4 / pos1 98.0 |
| r84_chain_s1     | 2 | 1 | **96.9%** | 97.0% | pos0 98.4 / pos1 98.4 |
| r84_chain_h3     | 3 | 0 | **95.2%** | 95.2% | pos0 98.8 / pos1 97.7 / pos2 98.8 |
| r84_chain_broken | 2 | 0 (broken) | **0.0%** | 0.16% | pos0 88.8 / **pos1 0.0** |

- **2-hop: 96.4% (seed 0), 96.9% (seed 1)** — matches the ~0.98² prediction; both hops ~98%.
- **3-hop: 95.2%**, each of the three hops 98–99% — composition depth extends cleanly.
- **Broken-chain control: 0.0%.** hop1 (A→B, which still exists) fires at 88.8%; hop2
  (the cut B→C link) collapses to **0.0%** → joint 0.0%. Cutting the chain kills exactly
  the second hop. This is the decisive control: the 96% is real chaining, not "return the
  most-recent value" (the leak that sank r70/r71).

## Why it works (vs the 7 latent negatives r68/r70–75/r82)
Latent chaining feeds a **retrieved representation** back as the next key; a value =
scale·(norm(last)+norm(embed)) is off the key manifold, so the next exact-match never
fires — seven straight negatives, including r82's hard-decode loop. Route A instead
**emits the intermediate as a real token** and re-enters the stack, so hop L+1 is an
ordinary single-hop lookup keyed by a genuine token embedding. Composition is a
working-memory op (retrieve → emit → re-query), not a within-forward latent thread.

The one remaining artifact was the store's, not the mechanism's: a token that is both a
key and a value (every chain intermediate) got its real slot clobbered by a spurious
`→pad` slot 50/50 (r83). `write_skip_pad` drops pad-valued writes, so the intermediate
keeps only its true `t1→t2` slot — hop2 45% → 98%, order-independently.

## What this establishes
CoT-over-memory on **one bounded shared store**: managed writes (one slot per distinct
key, O(facts), r76–79) + recency read (r67) + explicit token intermediates (r84). The
model composes over more facts than fit its context window, reading O(1) slots per hop.

## Caveat / next
These are **teacher-forced** (each intermediate is fed as ground truth at eval). The
honest end-to-end number needs an **autoregressive eval**: feed the model its OWN emitted
intermediate and measure joint recall. Expected ≈ the product of per-hop accuracies
(~0.96 at 2 hops) since the intermediates already sit at 98%, but it must be measured.
That is the immediate next step, then the paper section.

## Route B, for the record
r82 (loop-gate latent, hard-decode between hops) was a flat negative (0.2%, broken 0.2%)
— the 7th latent-multihop negative. Latent route parked; route A is the mechanism.
