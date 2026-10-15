# r105 — more heads do NOT close the 128-fact gap (null)

**Testing the first capacity lever from limitation #2: per-layer write at 128 facts with 4
attention heads (vs r104's 1) reaches 43.8% (best 44.1%) — essentially r104's 45.3%. More
heads is a null: the 4-head run tracked the 1-head trajectory step-for-step (2.8% at step
3.5k, matching r104) and landed at the same place. Attention-head count is not the 128-fact
bottleneck.**

## Numbers (cold bpe_recall, 128 facts, per-layer write token-key, d256/6L, batch 16, 10k)
| run | heads | recall (best = final) |
|-----|-------|------------------------|
| r104 spl_token_f128 | 1 | 45.3% |
| r105 spl_token_f128_h4 | 4 | **43.8%** (best 44.1%) |

Same 7.78M params (heads split the fixed d_model), so this isolates attention granularity, not
parameter capacity. See r106 for the curriculum and width levers, and the combined verdict:
none of heads / curriculum / width closes the 128-fact frontier (43.8 / 45.3 / 47.5%). Folded
with r106 into paper §limitations.
