# r85 — scale-up: width×depth, capacity frontier, multi-hop at scale, LM benchmark

**Four sub-studies across both boxes (resumable queue). Headlines: (1) multi-hop
CoT-over-memory SCALES to 32 facts; (2) the bounded store has a capacity frontier
between 64 and 128 facts; (3) shallow models cold-bootstrap the managed store, deep ones
do NOT — depth is the killer; (4) the memory block gives a small, real PG-19 LM gain.**

## 1. Multi-hop at scale (bpe_chain, warm from r79, 32 facts, M=1024)
| hops | recall (final) | best | per-hop char |
|------|----------------|------|--------------|
| 2 | **96.4%** | 96.6% | 98.4 / 98.0 |
| 3 | **95.2%** | 95.2% | 98.8 / 97.7 / 98.8 |

CoT-over-memory (r84) was shown at 16 facts; it holds at **32 facts** for both 2 and 3
hops, each hop still ~98%. Composition depth and fact count both scale.

## 2. Capacity frontier (bpe_recall, gated bounded store, warm from r79)
| facts | store M | recall (final) | best |
|-------|---------|----------------|------|
| 64  | 2048 | **95.9%** | 96.1% |
| 128 | 4096 | **26.1%** | 46.1% |

The bounded managed store holds full recall at 64 facts but does **not** converge at 128
in the 8k budget — it peaks ~46% then destabilises to 26%. A real frontier sits between
64 and 128 facts; whether it is the recurring under-training/bootstrapping deadlock (r25,
r66) or a genuine capacity wall of the single shared content is open (see the paper's
limitations).

## 3. Width vs depth — which shape cold-bootstraps the MANAGED store? (bpe_recall, gated, cold, 32 facts, M=1024, 12k)
| shape | layers | recall | read gate a0 |
|-------|--------|--------|--------------|
| d512_L2  | 2  | **96.7%** | open |
| d384_L3  | 3  | **95.3%** | open |
| d256_L6  | 6  | **0.3%**  | shut (~0.03) |
| d192_L12 | 12 | **0.2%**  | shut (~0.03) |

**Monotonic in depth: shallow (≤3 layers) cold-bootstraps the gated store, deep (≥6)
does not** — the read gate never opens, so the store is written (228 slots) but never
read. This is the first COLD test of the managed store (r76–79 were all warm-started),
and it says the write-op gate + store machinery cannot bootstrap from scratch in a deep
stack within 12k steps. Note this is a *bootstrap* result, not an *exploitation* one:
r87 (gate-free, same shapes) is running to test whether all shapes learn the plain store
(isolating the gate as the cause), and whether a curriculum removes the depth dependence.

## 4. Does the memory block help a real LM? (task lm, PG-19 char, memory on vs off, 8k)
| memory | val loss | ppl | per-segment (seg0→seg3) |
|--------|----------|-----|--------------------------|
| **on**  | **1.3382** | **3.81** | 1.358 → 1.339 → 1.329 → 1.326 |
| off | 1.3468 | 3.85 | 1.355 → 1.352 → 1.343 → 1.337 |

Memory ON beats OFF by **0.0086 nats** (ppl 3.81 vs 3.85), and the gain is concentrated
in **later segments**: with the store on, loss drops 0.032 nats from seg0 (empty store)
to seg3 (filled), versus 0.018 off. Segment 0 is ~equal (memory doesn't hurt when the
store is empty). So the block gives a small, real, store-filling-driven LM gain on
ordinary prose — the honest benchmark answer, consistent with r34's planted-fact-free
PG-19 result. It is a modest general-LM win, not the headline; the headline is targeted
recall/reasoning.

## Verdict
The finished construction scales along the axes that matter (multi-hop to 32 facts /
3 hops; LM gain real) and reveals two frontiers to chase: a 64–128-fact capacity wall,
and a depth-sensitivity in cold-bootstrapping the *managed* store. Follow-ups launched:
r86 (memory transplant — stateless weights / stateful store), r87 (gate-free width×depth
to separate bootstrap from exploitation).
