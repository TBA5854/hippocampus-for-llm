# r106 (+r105) — the 128-fact frontier is GENUINE: no capacity lever closes the gap per-layer write left

**After r104 showed per-layer write PUSHES the 128-fact wall (mean-write 0.3% → per-layer
45.3%) but only partially, we threw every capacity lever limitation #2 names at the gap.
None closes it. More heads: 43.8% (null). Curriculum warm-started from the converged
32-fact per-layer model: 45.3% — the EXACT cold ceiling (the warm-start only speeds the
climb, then converges to the same place). Width, d512 at 3.4× the parameters: 47.5% (+2.2
pts, replicated identically on two boxes). So per-layer write's ~45% at 128 facts is a real
frontier: a single shared store binding 128 distinct keys saturates near 45% regardless of
heads, curriculum, or width. 128 remains genuinely hard; per-layer write moves the wall, and
these levers do not remove it.**

## Numbers (cold bpe_recall unless noted, 128 facts, per-layer write, d256/6L, 10k steps; best = final)
| lever | config | recall | vs r104 cold 45.3% |
|-------|--------|--------|--------------------|
| (baseline, r104) | d256/6L, 1 head, cold | **45.3%** | — |
| more heads (r105) | 4 heads | **43.8%** (best 44.1%) | null (−1.5) |
| curriculum (r106) | warm from r103 32-fact per-layer ckpt | **45.3%** | null (0.0) |
| width (r106, box1) | d512/d\_mem 512 (26.8M params, 3.4×) | **47.5%** | +2.2 |
| width (r106, box2) | d512, replication | **47.5%** | +2.2 (identical) |
| (reference) mean-write @128 | r104 | 0.3% | the wall per-layer broke |
| (reference) per-layer @64 | r103/r104 | ~96% | the regime that still works |

## Reading
- **Heads: null.** 4 heads (43.8%) tracked the 1-head trajectory step-for-step (2.8% at step
  3.5k, matching r104) and landed at the same place. Attention-head count is not the bottleneck.
- **Curriculum: null ceiling, faster climb.** Warming 128 from the converged 32-fact per-layer
  model started at 32.2% (inheriting the 32-fact read path), dipped to ~8% as it overwrote that
  solution for the 128-fact store, then re-climbed — faster than cold (20% vs ~5% at step 4k) but
  to the IDENTICAL 45.3% ceiling. So the curriculum (which broke every earlier fact-count cliff,
  r25/r49/r66) does not break this one: those were under-training deadlocks a warm rung skipped;
  128 facts is not — the ceiling is the same whether cold or warm.
- **Width: a marginal, real bump.** d512 (3.4× params, d\_mem doubled) reached 47.5%, identically
  on both boxes — a genuine but tiny +2.2 pts. Doubling the per-slot dimension barely helps, so
  the limit is not per-slot content capacity either. (Its gate profile differs — layer 5 dominant
  — but the outcome is the same regime.)
- **The frontier is genuine.** Three independent levers (heads, curriculum, width) all land
  44–47.5% — per-layer write's ~45% ± noise. A single shared store binding 128 distinct keys
  saturates near 45%; per-layer write is what got it there from the mean-write's 0.3%, and none
  of these levers takes it toward the ~95% seen at 32–64 facts. 128 is a real capacity/stability
  frontier of the single shared store.

## Verdict
The capacity assault settles limitation #2: **per-layer write pushes the 128-fact frontier
(0.3% → 45.3%), and the obvious capacity levers — more heads, curriculum, 3.4× width — do not
close it (43.8/45.3/47.5%).** So the frontier is genuine, not a bootstrapping or budget
artifact. The remaining untested direction is structural (several shared stores / more read
heads on the store rather than attention heads), but the single-shared-store frontier at 128
distinct facts is now well characterised. r105 (heads) + r106 (curriculum, width) complete the
per-layer capacity investigation. Folded into paper §limitations.
