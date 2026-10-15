# r94 — multi-hop holds at 5 hops and 3-hop@64; frontier now 2–5 hops × 16–64 facts

**One more step out and it still holds: 5-hop @32 at 89.7% (per-hop ~98, ≈0.98⁵) and
3-hop @64 at 95.2%, both with broken-chain controls at 0.0% that kill only the cut hop.**

## Numbers (bpe_chain, warm r79, teacher-forced; per-hop = char position accuracy)
| run | recall | best | per-hop |
|-----|--------|------|---------|
| 5-hop @32 | **89.7%** | 89.8% | 97 / 98 / 98 / 98 / 98 |
| 5-hop @32, broken | **0.0%** | 0.0% | 97 / 98 / 98 / 98 / **0** |
| 3-hop @64 | **95.2%** | 95.2% | 99 / 98 / 99 |
| 3-hop @64, broken | **0.0%** | 0.2% | 95 / 94 / **0** |

## Reading
- **5 hops (89.7%)**: joint still tracks the per-hop product ($0.98^5\approx0.90$). The
  broken control fires the first four hops (97/98/98/98) and kills only the cut fifth
  (**0**) — deep composition, not a shortcut, at depth 5.
- **3 hops @64 facts (95.2%)**: chaining at 64 facts holds at 3 hops too (broken 95/94/0).

## Verdict
With r84/r85/r93, the multi-hop CoT-over-memory result is now mapped over **2–5 hops ×
16/32/64 facts**, every point with a broken-chain control at chance. Depth scales as the
per-hop product; fact count scales to the store's capacity. The teacher-forced numbers are
complete; the honest **autoregressive** figures (r95, feeding the model its own emitted
intermediate) are running. Folded into the paper.
