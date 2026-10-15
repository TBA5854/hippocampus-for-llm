# r69 — diagnose the multi-hop negative: RESULTS (cut short, key datapoint salvaged)

Stopped early on a pivot to the per-layer multistore idea (r70), but the stage-A
runs finished and answer the first and most important diagnostic question.

## Salvaged result: the hop-task format IS learnable at 1 hop
| arm | task | recall (best) |
|-----|------|---------------|
| box2 r69_hop1_retrieved | `bpe_hop --n-hops 1` (query A -> target B) | **98.8%** |
| box1 r69_hop1_token | `bpe_hop --n-hops 1 --hop-key token` | **95.5%** (still climbing when killed) |

## Interpretation
1-hop recall in the pad-separated triple format reaches ~99%, so the r68 negative
(all 2-hop arms ≈0%) is **not** a task-construction or warm-start bug — it is
specifically **2-hop chaining** that fails. This directly supports the r68
diagnosis: a raw retrieved value is not a clean exact key, so the second hop's
match does not fire.

## Not run (superseded by r70)
- hop2_curr (curriculum 2-hop warm from the 1-hop base) — box2 was mid-run.
- hop2_long (12k 2-hop) — box1 not reached.
r70 takes up the "does more structure fix chaining?" question from the multistore
angle instead (per-layer stores vs shared, matched curriculum). If r70's shared arm
also fails 2-hop under the curriculum, that closes the hop2_curr question too.
