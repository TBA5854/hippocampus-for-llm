# r89 — collision baseline confirms the 1/C draw; the WARM fine-key collapsed (confounded)

**Two halves. (1) The collision problem is real and quantified: an exact single-token key
lands on 1/C of C facts sharing a key token (C2 43.6%≈50%, C4 22.0%≈25%, C8 1.2%≈12.5%) —
r36's draw, now a clean baseline. (2) The fine (bigram) key, warm-started from r79,
COLLAPSED to 0.2% at every C — not "no lift" but total breakage, a warm-start confound
(a bigram key bolted onto r79's single-token-key read path breaks matching). The DG idea
is untested, not refuted; the clean cold test is r91.**

## Numbers (bpe_collision, 32 facts, M=512, warm from r79, 6k)
| C (1/C) | exact (baseline) | fine (bigram key) |
|---------|------------------|-------------------|
| 2 (50%)   | **43.6%** | **0.2%** |
| 4 (25%)   | **22.0%** | **0.2%** |
| 8 (12.5%) | **1.2%**  | **0.2%** |
| 4, scrambled-context control | — | **0.0%** |

## Reading
- **Baseline (exact) tracks 1/C**: the exact single-token key genuinely cannot separate
  facts sharing a key token — at C=8 it is essentially at chance (1.2%). This is the real
  collision axis behind "more than a context window" (keys collide constantly at scale),
  and the honest limit the exact-match design has been dodging (r36).
- **Fine key collapsed uniformly to 0.2%** at every C — including where it should have been
  trivially better (C=2). The uniformity across C is the tell: this is not the mechanism
  separating too weakly, it is the warm start breaking. The fine arm changes the layer-0
  key to a bigram; r79 was trained with single-token keys and a read path (ReZero gate,
  centring EMA, settling temperature) tuned to that key distribution, so swapping in bigram
  keys for 6k steps destabilised matching to chance. The exact arm, keeping single-token
  keys, warm-starts fine (hence the clean 1/C numbers). So the fine key was **not fairly
  tested**.

## Verdict
Collision baseline: SOLID (1/C, real problem). Fine key: NEGATIVE but CONFOUNDED by the
warm start — do not re-tune. The clean test trains exact and fine BOTH cold (from scratch),
so there is no single-token-key read path to break — **r91**, exactly as r87 corrected
r85's warm-only gate. If cold-fine >> cold-exact, DG separation works; if cold-fine also
collapses, the bigram key is genuinely incompatible with the frozen-identity exact match.
