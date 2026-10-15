# r96 — seed replication of the deep multi-hop points (robustness)

**Seed 1 reproduces seed 0 to the digit: 4-hop @32 = 91.6% (= r93 seed 0), 3-hop @64 =
95.2% (= r94 seed 0), broken-chain controls at chance. The deep multi-hop results are
seed-robust, not single-seed luck.**

## Numbers (bpe_chain, warm r79, seed 1)
| run | seed 1 | seed 0 (r93/r94) | broken (s1) |
|-----|--------|------------------|-------------|
| 4-hop @32 | **91.6%** | 91.6% | 0.2% |
| 3-hop @64 | **95.2%** | 95.2% | 0.0% |

## Verdict
The two deepest multi-hop points replicate exactly across seeds, with broken controls at
chance. Combined with r84's 2-hop seed replication and the r95 autoregressive confirmation,
the multi-hop CoT-over-memory result is seed-robust across the frontier. Nothing owed.
