# r64 — RESULTS: the CSA2 candidate-pool read works — O(1)-slot becomes O(1)-compute, reuse is free

The v0.5 core (plan: shared store + reuse). r62 left the read O(1)-**slots** but O(M)-**compute** (a full
q·K scan to find the 1–2 that matter). This builds the CSA2 fix: restrict the read to an index-selected
**candidate pool** — `token` (slots whose stored key-token equals the query's — exact for our
position-free layer-0 key) or `ann` (LSH bucket — approximate, for fuzzy keys) — and reuse the pool
across read layers. Offline `mech.persist pool` (eval, no retraining).

## Pool ablation at scale (r50_bpe64) — recall & mean candidates scored

| store | K win | slots | none (dense) | **token** | ann (LSH) |
|---|---|---|---|---|---|
| M=2048 | 15 | 1905 | 93.8% (1905) | **93.8% (1.3)** | 93.8% (153) |
| M=4096 | 31 | 3937 | 90.0% (3937) | **89.4% (1.5)** | 90.0% (338) |
| M=8192 | 63 | 8001 | 82.5% (8001) | **81.9% (1.9)** | 83.1% (642) |

(recall; parenthesis = mean candidates the read scores per query = the compute proxy.)

## Cross-layer reuse (r60_sh_f16_s2, a read@all model, 6 read layers)

| build | none | token | ann |
|---|---|---|---|
| per-layer rebuild | 100% (381) | 100% (1.0) | 100% (1.5) |
| **anchor-0 reuse** | 100% (381) | **100% (1.0)** | **100% (1.5)** |

## What it shows

1. **O(1)-slot becomes O(1)-compute (token index).** The token-hash returns exactly the ~1–2 slots that
   share the query's key-token, and reading only those matches the dense scan (within ~0.6 pt) at every
   store size — **~1.9 candidates out of 8001 at 63 windows.** The read no longer scales with M.

2. **ANN works for the fuzzy-key case.** LSH-bucket candidates (≈ M / 2^bits, so ~M/12 here) also match
   dense recall — the general index for when keys are not exact tokens, at a ~10× compute cut. Its pool
   grows ~linearly with M at fixed bits, so `lsh_bits` must scale with M to stay sub-linear (tuning note).

3. **Cross-layer reuse is FREE.** Building the pool once at an anchor layer and reusing it at the other
   5 read layers gives identical recall to rebuilding per layer (100% either way) — exactly CSA2's "few
   anchor layers, the rest reuse," confirmed. On a 6-read-layer model that is 6× fewer pool builds.

## Verdict
- **v0.5 core (candidate-pool read + cross-layer reuse):** ACHIEVED. Token-index makes the O(1)-slot
  read O(1)-compute (recall = dense, ~2 candidates of 8001); ANN generalises to fuzzy keys; reuse across
  layers is free. The store is now a shared KV that holds ≫ the window AND is retrieved cheaply.
- Guardrail: `read_pool="none"` reproduces r62 exactly (the dense column = r62), so no regression.

## Bug found & fixed (recorded)
Cross-layer reuse first collapsed to 0.6% — `pool_cache` was cleared per batch, so the query forward
reused the pool built during the last WRITE segment. Fix: clear the cache each forward (reuse is across
LAYERS within a forward, not across segments). After the fix, reuse = per-layer rebuild.

## Opens (deferred per plan)
- FP4-quantized value codes (CSA2's third axis).
- Train WITH the pooled read (only if a gap appears; none did at eval).
- Learned coarse code instead of random LSH; make the token/ANN index a real O(1)/O(log M) kernel (the
  probe measures candidates but still computes q·K densely for the recall number).
