# r103 — per-layer read/write into ONE shared store (the new default going forward)

## The change (user directive)
> "chnge to per layer r/w instead of last write first read for all exps going forward and shared store"

Stop doing **read-early / write-late** (one early read layer, one late write layer — or the
r92 variant: read every layer, write the *mean* of every layer = one slot/token). Instead
**every layer both reads from AND writes to the ONE shared store**. The store stays shared
(NOT per-layer stores — that's `per_layer_store`, regular-transformer stuff, dropped). Only
the read/write *placement* changes: reads at all layers (already supported) + a **per-layer
write** where each layer stores its OWN post-block value into the same store.

## Mechanism (implemented, smoked)
`cfg.shared_per_layer` + `cfg.shared_pl_key`:
- **Read** — the existing shared path with `read_layers = every block` (auto-set): a read head
  per layer, all querying the single `self.store`.
- **Write** — `commit()` loops every layer, writing that layer's `pending_values[i]` into the
  shared store. So each written token contributes **L slots** (one per layer), not one.
- **Key source (ablated):**
  - `token` — every layer's slot keyed by the shared layer-0 position-free token key. Exact-token
    match preserved; L slots per token share a key, differ in value (which depth's representation).
  - `residual` — each layer keyed by its own pre-read residual (transformer-native per-depth
    address). Tests whether the evolving stream can address the store (may lose exact match).

Smoke (CPU, 4-layer toy): both modes write **248 slots** vs the mean-write baseline's **62**
(= ~4× = per-layer writes landing at all 4 layers, grow absorbing the L×); baseline unchanged.

## Why this could help (and could hurt)
- **Help:** the store now holds every depth's view of a token, and every depth can read — a
  richer, more expressive memory than a single early-read / late-mean-write. Naturally subsumes
  the value-skip trick: layer 0's own slot IS the identity-bearing early value (r9), so no
  separate `--value-skip-layer` is needed. And L read layers give up to **L native hops**.
- **Hurt:** token mode makes L slots per token share one key — a built-in L-way collision the
  reader must disambiguate (by depth/recency); residual mode abandons the exact-token key that
  drives 100% recall (the raw-hidden-state cliff, cf. `center_keys`). Either could underperform
  the clean single-key baseline. That's the question.

## Design (train, cold, matched — bpe_recall, r92 hyperparameters)
Same model/task/steps for every arm; only the write scheme + key change.
- **baseline_mean** (both boxes) — read all 6 layers, write the MEAN of all 6 = one slot/token
  (`--write-layers 0..5 --value-skip-layer 0 --value-skip-norm`). The current default.
- **spl_token** (box1) — `--shared-per-layer --shared-pl-key token`.
- **spl_residual** (box2) — `--shared-per-layer --shared-pl-key residual`.
Baseline runs on BOTH boxes (cross-check); token on box1, residual on box2. Cold, 10k steps,
32 facts, M=512 seed + `--store-grow`.

## Predictions
1. **baseline_mean** reproduces r92-class recall (~100% best on 32 facts).
2. **spl_token** ≥ baseline if depth-values + more slots help; ≤ baseline if the L-way key
   collision dominates. Slots ≈ 6× baseline.
3. **spl_residual** < token (exact-token key lost), unless the frozen-identity query + centering
   still lands the residual near the right slot.
4. Whichever wins becomes the default for r104+. If baseline wins outright, report that per-layer
   *write* doesn't help on single-fact recall and move the test to multi-hop (its real promise:
   L layers = L hops), rather than re-tuning.

## Controls
- `--shuffle-store` at eval (wrong contents, same compute) → chance, per arm.
- baseline on both boxes = seed/box cross-check.

## Pass / fail
A clean verdict either way is a pass (a negative — "per-layer write doesn't beat mean-write on
recall" — is written up, not re-tuned). Fail = a crash or an unlearned (non-converged) arm that
confounds the comparison; matched 10k steps guards against that.
