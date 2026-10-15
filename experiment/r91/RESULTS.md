# r91 — COLD collision doesn't bootstrap (methodological negative); pivot to r92

**Trained from scratch, the collision task does not bootstrap at all: even the EXACT
baseline sits at 0.2% after 12k steps (vs 22%/1.2% when warm-started in r89). So the cold
approach cannot test the fine key — if exact-cold never reaches 1/C, fine-cold can't be
compared to it. Killed after both exact-cold arms confirmed the no-bootstrap; pivoting to
r92, which bootstraps the read path on the EASY recall task first, then warms collision
from it.**

## Numbers (bpe_collision, cold from scratch, 32 facts, M=512, 12k)
| run | final |
|-----|-------|
| r91_C4_exact_cold | **0.2%** (never transitioned; warm r89 was 22.0%) |
| r91_C8_exact_cold | **0.2%** (warm r89 was 1.2%) |
| fine-cold arms (C2/C4/C8), scramble | killed — cold task doesn't bootstrap, all would sit at ~0.2% |

## Reading
The collision task (facts `c k v`, C sharing each key) is too hard to cold-bootstrap in
12k: the read loop never establishes (recall flat at 0.2%, the pre-transition plateau),
unlike the plain recall task which cold-bootstraps by ~8k (r87). This mirrors r85's
cold+gate failure — a harder objective can't bootstrap the read/gate loop from scratch;
it needs a warm rung. So testing the fine key requires warm-starting from a model whose
read path is ALREADY bootstrapped AND compatible with the bigram key — which the r89 warm
start was not (single-token read path). r91 killed to reclaim both boxes for r92.

## Pivot (r92)
Bootstrap the bigram read path on the easy recall task first: (1) recall + fine-key COLD
(does the bigram key even bootstrap on an easy objective?), then (2) collision + fine-key
WARM from (1). Matched exact arm on box2 (recall exact cold → collision exact warm). If
(1) bootstraps and (2) >> the r89 exact baseline (22%), DG separation works; if (1) fails,
the bigram key is genuinely incompatible with the frozen-identity exact match.
