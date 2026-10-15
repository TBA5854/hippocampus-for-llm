# r70 — per-layer multistore vs shared store on multi-hop: RESULTS

> **RETRACTED by r71.** The 85% below was NOT chaining. r71's broken-chain control
> also scored 85.5% (cutting the chain changed nothing) and a seed-1 replication
> collapsed to 7.8%. The `bpe_hop` task leaked: the tail C is always the
> last-written value and the query is always the head, so "return the most-recent
> value" scores ~85% without reading the query. Multi-hop remains UNSOLVED. Kept
> for the record; read r71/RESULTS.md and the leak-proof r72 for the correction.

---


**Verdict: a per-layer MULTISTORE chains multi-hop (85% 2-hop); the single shared
store cannot (8.9%). Same task, same curriculum, `per_layer` the only difference.**
The user's instinct was right, and it reverses the earlier "per-layer is vanilla,
dropped" call: per-layer is the enabling structure for multi-hop.

## Numbers (best AND final; 2-hop = query head A, target tail C)

| box | store | stage A: 1-hop recall | stage B: **2-hop** |
|-----|-------|-----------------------|--------------------|
| box1 | **per-layer multistore** | 69.8% | **85.0%** (final 85.0) |
| box2 | shared (value-skip + recency) | 53.4% | **8.9%** (final 4.4) |

Trajectories (2-hop, warm from each box's own 1-hop base):
- multistore: 0 → 25.0 → 55.9 → **85.0** — a clean, steady climb; 2-hop is learned.
- shared:     0.3 → 2.7 → 5.6 → 4.4 — flat at chance; never fits the task.

## Why the gap is real, not a base-recall artifact
The 1-hop bases differ (pl 69.8 vs sh 53.4), but that cannot explain the 2-hop gap:
- multistore 2-hop (85.0) is **above** its own 1-hop base (69.8) — the model learns
  something the base did not have (chaining), and warm-start plus training lifts it.
- shared 2-hop (8.9) **collapses far below** its own base (53.4) — the 2-hop task
  destroys single-hop competence rather than extending it.
So per_layer is what enables chaining; the shared store actively fails it.

## Mechanism
Per-layer gives each depth its own store. Layer 0 is token-keyed and resolves the
first hop (A->B); deeper layers are keyed on the evolving residual (which now
carries B) against **their own** store, resolving the next hop (B->C). Distinct
stores mean the A->B and B->C slots never collide on one key — the exact collision
that sank the shared store (r68), where one token-keyed store returns the same slot
at every layer. "Lower layers do one hop, higher layers the next", exactly as
hypothesised.

## Placement in the story
Not one store or the other — **both, for different jobs**:
- **Shared store** — single-hop recall, long-context (>> window), sparse O(1) read,
  CSA2 pool/reuse/FP4 (r47-r65). The DeepSeek-style KV-sharing lives here.
- **Per-layer multistore** — multi-hop composition (r70). Chaining needs a distinct
  address space per depth so the residual can carry the running answer forward.

## Caveats / what r71 should harden
- **No per-layer broken-chain control yet.** The shared control at 8.9% is a strong
  negative control (same task, cannot leak), but a per-layer `--broken-chain` run
  (cut the last link -> must fall from 85%) would prove the 85% is genuine chaining,
  not a per-layer shortcut. r71.
- **Single seed**; replicate at seed 1.
- **3-hop** untested — does it chain A->B->C->D, or does depth run out?
- Per-layer 1-hop base (69.8%) is modest at 8k cold; a longer base might lift 2-hop
  further.
