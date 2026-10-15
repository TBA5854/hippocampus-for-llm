# r65 — CSA2 axis 3: how few bits does the store survive? (FP4 KV cache)

Fake-quantize the stored keys/values to N bits and read. Smoke: 4-bit = 100%, 2-bit = 0%.
box2: bit sweep x {dense, token pool} at 15/31/63 windows. box1: keys-only vs values-only (which limits?).
Completes the CSA2 axes (cross-layer reuse r64, candidate pool r64, quantization r65). Store bytes/slot = bits*d_mem/8.
