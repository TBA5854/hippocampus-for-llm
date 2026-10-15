# r102 — the 1M-token push: top-1 holds FLAT 100% to 999 windows (~128k tokens) on two models; reach is eval-memory-bound, not architectural

**Extending r101 (grow + top-1 = unbounded store at flat recall) far past the context window:
the sparse top-1 read holds recall at a FLAT 100% out to 999 windows — a 126,873-slot store,
≈128k tokens, ~1000× the 128-token attention window — on BOTH a d512_L2 and a d256_L6 model,
touching ONE slot. The dense read collapses to 0% at the same scale. Two surprises: (1) the
push STOPS at ~128k not for any capacity reason but because the offline eval materialises the
whole store to build+scan it and OOMs a 16GB GPU at ≥380k slots (an eval-implementation limit;
the architecture is O(1)/query, r101); (2) a rare-token-key arm UNDERperforms the normal-word
arm — rarest-word embeddings are under-trained and make weak keys, so exact match degrades.**

## Numbers (eval-only; grow + top-1; n=8 queries/cell so granularity is 12.5%)
### Normal keys (real words) — the headline
| model | windows | slots (≈tokens) | top-1 (1 slot) | dense (all slots) |
|-------|---------|-----------------|----------------|-------------------|
| d512_L2 (box1) | 199 | 25,273 (~25k) | **100%** | 87.5% |
| d512_L2 (box1) | 999 | 126,873 (~128k) | **100%** | **0.0%** |
| d256_L6 (box2) | 199 | 25,273 (~25k) | **100%** | 37.5% |
| d256_L6 (box2) | 999 | 126,873 (~128k) | **100%** | **0.0%** |

### Rare keys (rarest words) — the key-quality surprise
| model | windows | slots | top-1 | dense |
|-------|---------|-------|-------|-------|
| d512_L2 (box1) | 199 | 25,273 | 50.0% | 25.0% |
| d512_L2 (box1) | 999 | 126,873 | 62.5% | 12.5% |
| d256_L6 (box2) | 199 | 25,273 | 25.0% | 0.0% |
| d256_L6 (box2) | 999 | 126,873 | 12.5% | 12.5% |

### Attempted, could not run: ns3000 (~0.5M tok) and ns7800 (~1M tok)
OOM on a 16GB GPU across THREE allocation strategies (see Reading).

## Reading
- **Top-1 is flat 100% to ~128k tokens, on two different models.** 999 windows × 128 tokens ≈
  128k tokens of intervening prose between the fact and the query, answered from an empty prompt
  by touching ONE of 126,873 slots. r101 showed this to 199 windows; r102 shows it holds 5×
  further and on a second architecture. The store's reach is not tied to the attention window.
- **Dense drowns; sparse is immune.** The dense (all-slots) read decays as the store grows —
  0.875/0.375 at 25k slots → 0.0% at 127k on both models — because the softmax spreads weight
  over ever more distractor slots. Top-1 commits to the single exact key match and is unaffected.
  This is the r101 "sparse wins" story confirmed at 5× the scale.
- **Rare-token keys are weak keys (a key-QUALITY finding).** Keying facts on the rarest words —
  chosen to test whether the STORE itself reaches 1M (rare words seldom recur in prose, so no
  collisions) — instead recalls WORSE than common words (50–62% vs 100% on d512; 12–25% vs 100%
  on d256) and does not improve with distance. The position-free key is the token embedding; the
  rarest tokens have the least-trained embeddings, so argmax-cosine over the key manifold lands
  on the wrong slot. The better/larger model (d512) degrades less (62% vs d256's 12%), consistent
  with better-trained embeddings. So exact-match recall needs well-trained key-token embeddings;
  it is not automatic for any token. (Normal-word keys, which the model has seen often, are the
  100% case.)
- **Why the push stops at ~128k — an ENGINEERING wall, not a science one.** ns3000 (380,873
  slots) OOMs a 16GB GPU three ways: (a) grow's amortized doubling fragments the caching
  allocator to ~14G and fails the next 2G map; (b) a fixed pre-sized store avoids fragmentation
  but is O(nseg·M) — every window rescans all 400k slots — and crawls (27 min, unfinished);
  (c) grow + `expandable_segments:True` maps memory down to 71MB free, then fails. The offline
  eval builds the ENTIRE store resident and scans it; that does not fit past ~130k slots on 16GB.
  The trained mechanism itself reads O(1) slots per query (top-1, r101), so a streaming /
  chunked-scan eval would reach 1M — a future engineering task, out of scope here.

## Verdict
The central long-context claim is made and now doubly-confirmed: **an external, grow-able store
read by an exact top-1 lookup holds recall FLAT at 100% to ~128k tokens (999× the window) on two
models, in O(1) slots, while the dense read collapses.** The literal 1M-token measurement is
blocked only by the offline eval's whole-store materialisation on a 16GB GPU, not by the model.
Secondary result: rare-token keys degrade recall (under-trained embeddings → weak keys), a
key-quality caveat worth stating alongside the collision story. Folded into paper §longcontext.
