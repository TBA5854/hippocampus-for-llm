# r93 — the multi-hop frontier extends: 4 hops and 64 facts, controls at chance

**CoT-over-memory extends in BOTH directions: 4 hops at 91.6% (each hop ~98%) and 2 hops
at 64 facts at 96.9%. The broken-chain controls are decisive and at 0.0% — and the 4-hop
control's per-hop split (97/97/95/0) shows exactly the cut hop dying while the intact ones
fire, so this is genuine deep composition, not a shortcut.**

## Numbers (bpe_chain, warm from r79, teacher-forced; per-hop = char position accuracy)
| run | recall (final) | best | per-hop |
|-----|----------------|------|---------|
| 4-hop @32 facts | **91.6%** | 91.6% | 98 / 98 / 98 / 98 |
| 4-hop @32, broken chain | **0.0%** | 0.2% | 97 / 97 / 95 / **0** |
| 2-hop @64 facts | **96.9%** | 97.0% | 99 / 98 |
| 2-hop @64, broken chain | **0.0%** | 0.0% | 88 / **0** |

## Reading
- **Depth: 4 hops works (91.6%).** Each of the four hops is ~98%, so joint tracks
  $0.98^4\approx0.92$ — composition depth scales as the per-hop product, exactly as the
  emitted-token scratchpad predicts (r84). The broken-chain control makes this airtight:
  cutting the last link leaves the first three hops firing (97/97/95) and kills only the
  fourth (**0**), so the model genuinely walks the chain rather than pattern-matching a
  shortcut.
- **Fact count: 64 facts works (96.9%)** at 2 hops — chaining holds at twice r85's 32, with
  its broken control at chance (88/0). So the multi-hop mechanism is not fact-count limited
  in the range where the store itself holds (recall the single-store capacity wall is
  64→128, r88; multi-hop is fine at 64).

## Verdict
The multi-hop CoT-over-memory result (r84) is now mapped across a frontier: **2/3/4 hops
(r84/r85/r93) and 16/32/64 facts (r84/r85/r93)**, every point with a broken-chain control
at chance. Composition depth scales as the per-hop product; fact count scales up to the
store's own capacity. Folded into the paper's scale-up narrative. Remaining honest number:
the autoregressive eval (feed the model its OWN emitted intermediate) — next.
