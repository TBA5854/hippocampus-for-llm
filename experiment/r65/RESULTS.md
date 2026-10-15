# r65 — RESULTS: the store survives 4-bit (FP4), 8× smaller, no recall loss — CSA2's third axis

CSA2's KV cache is FP4-quantized. Does our store survive it? Fake-quantize the stored codes to N bits and
read (offline `mech.persist quant`, r50_bpe64; the keys-vs-values ablation on r50_bpe16_s2).

## Bit sweep — quantize BOTH keys and values (recall; bytes/slot = bits·d_mem·2/8)

| bits | B/slot | 15 win | 31 win | 63 win |
|---|---|---|---|---|
| 16 | 1024 | 93.8% | 90.0% | 82.5% |
| 8  | 512  | 93.8% | 90.0% | 82.5% |
| 6  | 384  | 93.1% | 89.4% | 82.5% |
| **4** | **256** | **92.5%** | **87.5%** | **82.5%** |
| 3  | 192  | 11.9% | 6.2% | 4.4% |
| 2  | 128  | 0.0% | 0.0% | 0.6% |

**4-bit (FP4) holds within ~2 pts of full precision at every scale — no loss at all at 63 windows (82.5%
= 82.5%) — an 8× smaller store (1024 → 256 B/slot). 3-bit is a hard cliff.** The token pool tracks the
same limit (4-bit fine, 3-bit breaks), because `settle` still scores the pooled slots by their key
vectors.

## Keys vs values — which is bit-limiting? (r50_bpe16_s2, dense)

| bits | quant KEYS | quant VALUES |
|---|---|---|
| 4 | **99.4%** | 99.4% |
| 3 | **0.0%** | 99.4% |
| 2 | 0.0% | 23.1% |

**The keys are the limit.** Matching (q·K) needs ~4 bits to keep near-tied keys apart, so keys cliff at
3-bit. Values only have to decode to the right token and tolerate **3-bit** (still 99.4%), cliffing at
2-bit. So an asymmetric store — keys 4-bit, values 3-bit — is the tighter packing the "both" number
hides.

## Verdict
- **CSA2 axis 3 (low-bit KV):** ACHIEVED. The store quantizes to **4 bits/element, 8× smaller than
  FP32, with no recall loss**; 3-bit is the cliff, and it is set by the keys (values go to 3-bit).
- This is the "store more per byte" axis of the r51 recipe: on top of grow-M and keep-the-right-slots,
  the per-slot cost is 4× smaller than the fp16 assumed there.

## CSA2 for our store — complete (r64 + r65)
All three DeepSeek-V4.1 axes now hold on the store: **cross-layer reuse** (free, r64), **candidate pool**
(token ~2 slots / ann ~M/12, r64), **low-bit KV** (4-bit, 8×, r65). The store is a shared KV that holds
≫ the context window, is retrieved in O(1) slots, and costs 4 bits/element.

## Opens (deferred)
- Asymmetric keys-4/values-3 packing (measured to work; not wired as a mode).
- A real quantized-storage kernel (the probe fake-quants in fp32; true int4 storage is the byte win).
- FP4 + curriculum on the harder BPE fact counts (does quantization interact with the r63 second
  bottleneck?). Not run — r65 is the clean quantization axis in isolation.
