# r103 — per-layer read/write beats read-all/write-mean: the new default is a +16pt win (with the exact token key)

**Switching from "last write, first read" (read every layer, write the MEAN of every layer =
one slot/token) to PER-LAYER write (each layer stores its OWN value = L slots/token) into the
ONE shared store, addressed by the exact token key, lifts cold bpe_recall from 80.8% to
95.5%/98.9% (two seeds) — a +14.7/+18.1 point gain — and bootstraps earlier (step 4–6.5k vs
7.5k). The residual-key control, which has the SAME 6× slot count but keys each layer by its
evolving residual instead of the token, collapses to 0.2% — so the win is per-depth VALUES
addressed by the EXACT TOKEN KEY, not raw slot capacity. The user's reframe ("per layer r/w
instead of last write first read, shared store") is not just a reframe: it measurably improves
recall. Adopted as the default going forward.**

## Numbers (cold bpe_recall, 32 facts, d256/6L, 10k steps, best = final)
| arm | key | slots/token | recall (best=final) | bootstrap step |
|-----|-----|-------------|---------------------|----------------|
| baseline_mean (read-all / write-mean) | token | 1 (~508) | **80.8%** (both seeds/boxes) | 7500 |
| **spl_token** seed0 | token | 6 (~3048) | **95.5%** | 6500 |
| **spl_token** seed1 | token | 6 (~3048) | **98.9%** | 4000 |
| spl_residual seed_ctrl | residual | 6 (~3048) | **0.2%** | never |

baseline_mean reproduces IDENTICALLY across box1 and box2 (80.8%, boot@7500) — clean replication.

## Reading
- **Per-layer write helps, a lot.** Storing each layer's own post-block state as its own slot
  (L=6 slots/token) instead of averaging all six into one slot lifts recall 80.8% → ~97%. The
  mean-write throws away depth-specific content (one averaged vector must carry the whole
  key→value binding); per-layer write keeps each depth's view, and the read head at each layer
  can recover the representation best suited to it. Gates open EVENLY across all six layers
  (0.07–0.27) under per-layer write, vs layer-0-dominated for the mean-write — every depth uses
  its own slot.
- **The exact token key is essential (the residual control).** spl_residual has the identical
  6×-slot store but keys each slot by that layer's evolving residual; it never learns (0.2%, read
  gates never open — nothing matches, so there is no gradient to open the gate). So the +16pt win
  is NOT the extra slots (residual has them too) — it is per-depth values that stay ADDRESSABLE
  because the key is still the exact position-free token. Removing exact addressing destroys it,
  exactly as the raw-hidden-state key cliff (config `center_keys`) predicts.
- **Faster bootstrap.** spl_token crosses its recall jump at step 4000–6500 vs the baseline's
  7500 — more, richer, per-depth slots give the read path an earlier foothold.
- **Open mechanism (not chased here — mech-interp deferred).** In token mode the 6 slots for a
  token share the SAME key token, so a token-key query matches all six; the per-layer read head
  must be selecting among them (by value fit / recency). WHY it selects correctly is a mechanistic
  question left for the dedicated interp pass; the behavioural result stands.

## Verdict
**Per-layer read/write into one shared store, token-keyed, is the new default** — a matched,
2-seed, +16pt improvement over read-all/write-mean on cold recall, with a control that isolates
the cause (per-depth values + exact token addressing, not capacity). The residual-key variant is
a dead end. Next (r104): the real promise of per-layer r/w is MULTI-HOP — L read layers = up to L
native hops — which needs the per-layer reads to CHAIN (layer i's retrieval → layer i+1's query);
small code change, then test vs the emitted-token CoT (r84/r95). Folded into paper §method.
