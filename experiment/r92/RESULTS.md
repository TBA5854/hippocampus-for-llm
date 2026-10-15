# r92 — the fine (bigram) key is incompatible with exact match: DG-via-bigram is DEAD

**Trained cold on the EASY recall task, the exact key bootstraps to 84.7% but the bigram
(fine) key stays at 0.0% with the read gate shut — same settings, only the key differs. So
the fine key's failure is not the r89 warm-start confound; the bigram key genuinely cannot
bootstrap the frozen-identity exact-match read. Adding a fine term to the key breaks the
exact match everything depends on. The fine-key / DG-via-bigram approach is closed.**

## Numbers (cold from scratch, bpe_recall then bpe_collision warm, 32 facts, M=512)
| run | final | note |
|-----|-------|------|
| r92_recall_exact_cold (control) | **84.7%** | exact key bootstraps cold (= r87) |
| r92_recall_fine_cold (CRUX)     | **0.0%**  | bigram key never bootstraps, read gate shut (~0.11) |
| r92_col_C4_exact_warm | ~22% (→ 1/C, matched baseline) | warm from the 84.7% recall model |
| r92_col_C4_fine_warm  | 0.0% | warm from the 0% recall model — nothing to warm from |

## Reading
- The crux is the recall A/B: exact 84.7% vs fine 0.0%, identical but for the key. The
  bigram key emb(k)+α·emb(prev) changes the key distribution the frozen-identity query must
  match; the read gate never opens (α₀ ~0.11 vs the exact run's 0.53), so the store is
  written but never read. This is the same failure mode as a contextualised key (r42) and a
  random-projection key (r31): **anything that moves the key off the raw token embedding
  breaks the exact match**, which is the load-bearing property of the whole design.
- So the three attempts to separate collisions by enriching the KEY have all failed:
  contextual key (r42), and now the bigram fine key cold (r92) and warm (r89). The exact
  single-token key cannot be extended in place.

## Verdict & pivot
Fine-key / DG-via-bigram: **DEAD** (write-up, not re-tuned). The collision limit (1/C, r89)
stands. The lesson: don't add a fine term to the key — that breaks exact match. The right
direction for collision/fuzzy recall is to **relax the COARSE term at READ**: make the
existing LSH/ann candidate pool (r64) actually carry recall so a near-key retrieves, rather
than changing the stored key. That (and the autoregressive multi-hop eval) are the next
threads; the fine key is dropped. Folded into the paper limitations.
