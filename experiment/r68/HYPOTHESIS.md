# r68 — multi-hop through the ONE shared store

Reads never chain today: each read layer queries the same layer-0 token key, so
the store answers "where is A" but not "A -> B -> C". r68 makes read layer L's
retrieval the query for read layer L+1 against the SAME shared store — one hop per
layer, the direct extension of r64's cross-layer reuse. value_skip_layer 0 makes a
stored value carry the value token's embedding, so a retrieved value is usable as
the next hop's key. Task `bpe_hop`: chain A->B, B->C stated once; query A, target C
(2-hop). Chain facts are written in order + recency read (r67) resolves the
spurious value_shift slot on each intermediate. Warm from r66_mlcold_f16 (99.7%
@16, ml placement). box2, n_facts 16, seed 0.

Arms (all --recency-read, 2-hop):
- **hop_retrieved** — `--hop-key retrieved`: retrieved value feeds the next hop.
  Prediction: 2-hop recall well above the broken-chain control.
- **hop_token** (control) — `--hop-key token`: every layer re-reads the head A, no
  chaining. Prediction: can reach B but NOT C -> 2-hop fails (~chance).
- **hop_broken** (control) — retrieved + `--broken-chain`: last link cut, head
  cannot reach tail. Must fail even with chaining -> isolates that success is real
  chaining, not leakage.
- **hop_residual** — `--hop-key residual`: evolved residual as the next key
  (transformer-native). Riskier: gives up the exact-token key; ablation vs
  retrieved.

Pass: hop_retrieved >> hop_token and >> hop_broken on 2-hop recall. If retrieved
also fails, the shared store cannot chain single-lookup reads and the thalamic
relay / dedup is needed (a result either way). Report best AND final.
