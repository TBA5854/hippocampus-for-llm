# r111 — RESULTS: the literal 1M on a second model + valid controls + rare-key at 1M

**Verdict: the million-token reach generalises across models, the retrieval is genuine
(controls at chance), and the rare-key quality effect persists undiminished at 1M.**
Completes the long-context story (r108 measured 1M on d512; r111 confirms it is not
particular to one model and is real, not a filler artifact).

## Setup
`mech.persist sparse`, top-1 read, fixed pre-sized store, **batch 1** for the 1M cells
(top-1, single slot touched), **batch 2** for the scrambled control (batch-1 shuffle is
degenerate — see below). Store built with the r108 eval-mode in-place O(S) write, so a
~10^6-slot store is buildable in bounded memory. Facts: 32, seg_len 128.

## Results

### 1. Cross-model literal 1M (box2, d256 / 6-layer, r92 ckpt)
| nseg | slots | ~tokens | top-1 recall |
|---|---|---|---|
| 1000 | 126,873 | ~128k | **100.0%** |
| 3000 | 380,873 | ~384k | **100.0%** |
| 7800 | **990,473** | **~998k ≈ 1M** | **100.0%** |

r108 measured the literal 1M on d512/2-layer; r111 reproduces it at **100%** on the
**d256/6-layer** model — so the reach is a property of the external store, not of one
architecture. (r102 had already shown the ~128k point on both models; r111 pushes the
second model to the full million, touching **one** slot.)

### 2. Retrieval is genuine — matched scrambled control (box1, d512, batch 2)
| nseg | slots | real store | scrambled (`--shuffle`) |
|---|---|---|---|
| 1000 | 126,873 | 100% | **0.0%** |
| 3000 | 380,873 | 100% | **0.0%** |

Reading a *neighbour's* store instead of the query's own drops recall to **chance at
128k and 384k**, so the 100% is real content-addressed retrieval, not leakage from the
prompt or the filler.

> **Control caveat (do not repeat):** the scrambled control MUST run at batch ≥ 2.
> `shuffle_store` permutes stores *across the batch*; at batch 1 there is no neighbour to
> swap with, so the read falls back to the query's own store and the "control" scores
> ~100% (a degenerate non-control, not a genuineness failure). First batch-1 attempt gave
> 100%; the batch-2 rerun gives the correct 0.0%.

### 3. Rare-key quality effect persists at 1M (box1, d512, batch 1, n=4)
| nseg | slots | rare-key top-1 | normal-key |
|---|---|---|---|
| 3000 | 380,873 | 50% | 100% |
| 7800 | **990,473** | **75%** | 100% |

Keying facts on the **rarest** words (whose embeddings are the least trained) holds
top-1 to **50–75%** even at a million tokens, vs 100% for normal-word keys. This is the
r102 **key-quality** effect — under-trained rare-token embeddings make weak keys — and it
neither closes nor worsens catastrophically at scale; it is a property of the key
embedding, not of store size. (Small sample, n=4 per cell; qualitative.)

## Takeaways for the paper
- The **1M is measured on two models** (d512 r108 + d256 r111), both 100%, one slot.
- **Controls at 0.0%** at 128k and 384k confirm genuine long-context retrieval.
- The rare-key caveat is a **key-quality**, not a capacity, limit — persists at 1M.
- Fold into §longcontext (strengthen the r108 1M claim: cross-model + controls).

## Sequel
r112 (passkey / needle-in-a-haystack) then isolates the **needle-depth** axis and
uncovers a write-order/primacy decay — a separate, sharper result written up in r112/r113.
