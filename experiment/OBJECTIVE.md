# Objective reached — 2026-09-13

**Objective:** a memory block inside the stack — read at the first layer, write at the last block before
the head — that makes the model stateful: it answers from its own store with nothing in the prompt, and
the store persists.

**Done criteria (set before r1):** >= 8 facts out of context, store persisted and reloaded, matched
controls at chance. All met, on one configuration.

## The configuration

`mech/` TinyGPT, 6 blocks, d=256, trained from scratch on 8-fact MQAR (facts in segment 0, two distractor
segments, query in segment 3 — earlier segments never in context).

```
--key-layer 0 --read-layers 0     key and query from block 0's input (token + position embedding)
--write-layer 5 --value-shift 0   value = last block's state at the key position ("what comes next")
--query-init identity             query and key start in the same space
--memory-size 512 --beta-grad 4 --dg-bypass
```

## Evidence

| claim | result | where |
|---|---|---|
| 8-fact recall, facts out of context | **100.0% on seeds 0, 1, 2** | r5_k0, r5_k0_s1, r6_f8_s2 |
| store disabled (HC off) | 2.5% | r4_f8_off |
| scrambled store (another example's memories) | 2.7% | r6_f8_shuf |
| chance | 3.1% | 1/32 |
| store saved, loaded in a **fresh process**, query segment has no facts | **100.0%** == in-process 100.0% (n=320) | r6/offline/read.json |
| fresh process, **mismatched** store file | 3.4% | r6/offline/read_mismatch.json |
| store loaded by a different model | **refused** (weight fingerprint) | persist.py, seed-1 model |

Correct-slot reads are 100% from step 250 to the end of training on every seed; misses before lift-off are
all the gate not yet open, never a wrong slot.

## How it got here (one line per round)

- **r1** writing at the last block with keys from that block collapses (70.5% -> 44.4%).
- **r2** the last block encodes the *next* token, so it pairs by prediction; layer-1 keys make content
  transfer near-perfect but a random query init pins matching at ~1/2.
- **r3** identity query init: 8 facts 83.9% (old architecture: chance) — on one seed.
- **r4** seeds 1-2 at chance: layer-1 keys drift as block 0 trains. Persistence across processes works.
  Every miss is another fact's key.
- **r5** layer-0 keys have nothing to drift: 100% on every seed. A gradient floor does not help (3.9%).
- **r6** seed 2 100%, scrambled control 2.7%; facts 16/32 and a 4x store running.

## Update — facts seen once (r7-r9, 2026-09-14)

The prediction-caching limit below is now **measured and partly overcome**. On single-occurrence facts
(`--no-tile`), with position-free layer-0 keys and the query frozen at identity:

| value written from | recall |
|---|---|
| block 0 (observed token) | **100.0%** |
| block 3 | 25.0% |
| last block, observed position | 2.3% |
| last block, prediction (the r5 config) | 2.3% |

Matching is 100% in all four, so the column is purely how much of what was *seen* survives at that depth. A store
written from block 0 holds facts seen once; one written from the last block does not. Getting there also required
two fixes found on the way: position-free keys (r7: position embedding hijacked matching) and a frozen query (r8: the
shared learned query was steered away from facts by text positions).

**r10 closes both halves:**

| on single-occurrence facts | result |
|---|---|
| block-0 store, seeds 0 / 1 | 100% / 100% |
| block-0 store, scrambled | 2.3% |
| block-0 store, 16 / 32 facts | 100% / 100% (no fact-count cliff) |
| **last-block write + block-0 skip** | **93.9%** (still rising at 4000 steps) |
| **same, store reloaded in a fresh process** | **94.06%** == in-process; mismatched store 3.4% |

So the original objective's placement — read at the first layer, write at the last block — now stores and persists
facts the model saw exactly once, given a skip path from block 0 into the value.

**r11 makes it robust** (8000 steps): seed 0 **100%**, seed 1 **100%**, 32 facts **100%**, scrambled store **2.8%**.
Its store, saved and reloaded in a fresh process: **100% == in-process**, mismatched store 3.4% (n=320).
See `r11/RESULTS.md` for the full list of ingredients and which failure each one fixed.

**Scaling the last-block write (r16-r18, TinyStories):** 1x 100%; 2x needs the halves normalised (`--value-skip-norm`:
99.4%; plain 28.1%); 4x has not lifted off from scratch (plain stopped at 5000, normalised at chance through 12000). The
block-0 store alone holds 100% at 1x/2x/4x.

**Store-size curriculum (r19):** warm-starting the 4x run from the 2x checkpoint (`--init`) reaches **100%** (99.1% by
step 750), with the matched scrambled control at **2.8%**. Its 4x store (1,905 slots per example) saved and reloaded in
a fresh process gives **100% == in-process**, and a mismatched store 2.5% (n=40). So the objective placement is established at
1x, 2x, 4x and **8x** (r20: 4,064 slots per example, 100% from step 500, scrambled control at chance, reloaded store
100% in a fresh process). Large stores fail from scratch because the model never starts using the read (lift-off), not
because the store lacks capacity: the transferred read keeps mass 0.96-0.97 on the correct slot from the first eval,
and each doubling warm-starts more easily than the last (79.7% -> 96.2% at the first eval, 4x -> 8x).

## Consolidation — sleep as the eviction mechanism (r26-r27, 2026-09-17)

The store is a **readable transcript**: position-free layer-0 keys are token embeddings, so nearest neighbour recovers
them exactly (100% at 381 and at 3,937 slots), and slot i+1's key is slot i's value token — so replay needs no decoder
and no token buffer (r26). Values themselves do not decode (0% on fact symbols): they are a normalised mixture of the
last block and block 0.

On that, the full loop runs (r27): decode -> replay with reads off -> verify with memory off and an empty prompt ->
free what verifies. Consolidation goes **0% -> 100%**, all 381 slots reclaimable, and LM loss *improves* (0.961 ->
0.833) when sleep interleaves general rehearsal. Two findings worth carrying:

- **Rehearsal is the protection that matters; freezing the key space backfires.** Pinning block 0 keeps old keys valid
  but forces adaptation into the later blocks, decalibrating the write path from the read path: new writes fall to 25%
  against 75% when the model is allowed to move coherently.
- **Weight drift has a cliff.** Up to ~120 the store survives sleep (88-100% readable); at ~418 it is unreadable (0%),
  keys frozen or not.

**r28 closes the write-path cost and names the real tension.** Rehearsing the *memory task* during sleep restores new
writes to **100%** with consolidation still at 100% and LM loss improved, and the whole loop holds at the **8x store**
(3,937 slots). But the key space cannot be in two places at once: stored keys are token embeddings, so reading them
back wants the embedding where it was, while writing new memories correctly wants it where the network now is. Freeze
it and new writes fall to 25%; rehearse into it and the old store falls to 38%.

The architecture resolves this itself: because the store decodes to exact token ids, an episode that did not
consolidate can be **re-encoded from its own transcript** under the new weights (100%). So the loop is: decode ->
sleep (LM + memory rehearsal, reads off) -> verify with memory off -> free what verifies -> re-encode the rest.

Open: consolidation is well-posed only **one user at a time** (a batch of contradictory fact sets consolidates at
3.1%), and every measurement so far is a **single** wake/sleep cycle — steady-state behaviour over many cycles is
r29.

## Compression — by write RATE, not slot width (r30, 2026-09-17)

The plan meant `d_mem` by "compression". The bigger lever is **how many slots an episode spends**: causal attention
means position t's state already summarises everything <= t, so one slot per token stores the episode many times over.

Writing only the **top 25% of positions by the model's own next-token NLL** reaches **100.0%** against a 99.7% baseline
that writes every token — 4x fewer slots, no loss, and it *learns faster* (99% by step 6750 vs 8000 for the baseline).
Position-based skipping at the same budget collapses (stride 4: 0.5%), so the win is not fewer slots but **less
redundant** ones: predictable tokens make near-duplicate keys, which is r5's correlated-key failure seen from the write
side. One vector per episode (the v0.1 design) is at chance, as r9 predicted.

Fact-key coverage stays at 100% at both working rates, so **replay for sleep survives compression**. 12.5% fails
(3.6%) with coverage still at 90.6% — a lift-off failure at a viable store, not a capacity limit.

**Slot width does NOT compress (r31d).** `d_mem` 64 and 32 both finish at chance with `slot@1` at **0.0%** — a frozen
random projection destroys the exact-token matching that position-free layer-0 keys provide, which is the same
property r26 relies on to decode the store and sleep relies on to replay it. The two axes are not interchangeable:
write rate is available, slot width is not, and buying width back would need learned/contextual keys — the direction
r4 found drifts, and which would cost exact decodability.

**Seed check (r30, seed 1):** the 25% configuration reproduces exactly (100.0% on both seeds); the full-write baseline
collapses to 27.8% on seed 1, still climbing at step 8000. So the redundant store is the *fragile* one, and
compression is not a trade — it is what makes the read learnable on this task.

## Episodes and eviction (r31, 2026-09-17)

The store was a ring buffer from r1 to r30, which never mattered because every round sized it to hold everything.
Under real overflow the ring is catastrophic: facts are written first, so they are always the oldest slots, and an
overflowing ring evicts exactly what the task needs. Measured at chance (0.5%) in every overflowing configuration.

**Episode-aware eviction** — slots owned by episodes, spend free slots, then INCOMPLETE episodes oldest-first, and a
complete episode only when nothing else remains — turns the same stores into working ones: **0.5% -> 90.6%** at a
64-slot store with compressed episodes, **0.5% -> 82.5%** at 288 slots at the full write rate, with the LM improving
alongside (ppl 897 -> 3.8 / 7.4). That is the rule from the original design conversation, now measured.

Completeness is not a label: r26/r27 made it mechanical (a complete episode has a replay target and can be
consolidated; an open one cannot), so the same distinction governs eviction and sleep.

**Sizing rule:** the policy needs headroom — room for the complete episodes plus at least one replaceable one. At a
store too small for that, ring and episode policies tie at chance, since the only thing left to evict is what we are
trying to protect. Capacity for "a day's worth" must therefore be budgeted per *episode*, with one episode of slack.

## What this does NOT show

- ~~Facts the model has not already predicted.~~ **Overcome in r9-r11** (see the update above): the value now
  carries the observed token, and facts seen once are stored at 100%. Original note, kept for the record —
  **this was the tightest limit, found in r6.** The stored value
  is the last block's state at the key position, with shift 0 — the model's *prediction* "next is `v_j`".
  So the store can only hold what the model could already predict from context when it wrote. MQAR tiles
  the facts through segment 0 (8 facts = 8 repeats), so induction makes `v_j` predictable and the value
  carries it. At 16 facts (4 repeats), `r6_f16` sat at chance for 2000+ steps while matching was 99%, and
  its segment-0 loss held at **3.44 ≈ ln 32** — uniform over the symbols, no induction yet, nothing
  useful in the value. The implication: **a fact seen once, the normal case for real memory, would not be
  storable in this configuration.** It caches predictions, not observations. Strongly indicated, not yet
  tested directly (needs a single-occurrence-fact variant of the task); the fix direction is a value
  that carries what was *observed* — e.g. the value half from an identity-bearing layer, or shift 1 on a
  layer that still holds token identity.

- **Retrieval by meaning.** A layer-0 key is token + position embedding, so matching is close to exact
  token lookup. The mechanism — store in the block, written at the last layer, read at the first,
  persisted to disk, gated by a learned read gate — is demonstrated. Retrieving a fact from a paraphrased
  or contextually cued query is not; that needs contextual keys, which drifted in r4. r5_bg16 (a steeper
  surrogate lets matching *learn*, 21% -> 70%) is the lead.
- **Natural language.** MQAR is a synthetic key->value probe. It is the standard test for this capability,
  but "removes the need for RAG" is a claim about real text that has not been tested.
- **Scale.** 6 blocks, d=256. Fact count is no longer the limit for the observation store (32 single-occurrence facts
  at 100%, r10/r11); the key vocabulary (32 symbols) is. Store size: observation store on facts seen once reaches
  **100% at 2x (896 slots, r13) and 100% at 4x (1,920 slots, r15, on TinyStories)** at effective batch 32 — larger
  stores lift off later but are not capped. On TinyStories the language model no longer memorises the corpus (val LM
  loss 0.73 vs train 0.70; tinyshakespeare at 4x was 2.59 vs 0.35). Earlier note: r6: 16 and 32 facts fail on content (above), with
  matching still ~100%. A 4x store (1,920 slots, 14 distractor segments) kept matching at 100% but its
  recall is untested: it needed batch 8 to fit, induction stalled (segment-0 loss 1.93 vs 0.43 floor), and
  the value was empty for the same reason. Memory scales with segments x store size, since each segment's
  `[B, S, M]` read tensors live until backward.
- **The write gate.** Writes are ungated per-token (the gate is bypassed in per-token mode), by decision.


## Extended to real language — 2026-09-18 (r34-r48)

The original objective was met on MQAR (synthetic single-token symbols outside the text vocabulary). The natural
question — does it work when facts are stated in real language? — was pursued r34-r48 and is now answered YES.

**Char-level facts in text (r35-r46):** a fact `ab:cd` written in prose, queried out of context. Works, but the
char-level layout makes every key a MULTI-token key addressed via the shared `:` delimiter, so recall is a
1/n_facts draw (r36) and declines with fact count: 4f 90%, 8f 79%, 12f 62%, 13f 55% (r44/r46). The decline is that
multi-token-key tax compounding, plus the r25 bootstrapping deadlock (escaped by a curriculum).

**BPE token-level facts (r47-r48):** keys and values are SINGLE real word-tokens in the PG-19 vocabulary (8192),
competing with the prose embedding space. This removes the addressing tax (r47: 8f cold 49.8% vs char 10.3%,
char==exact) and, with the curriculum, reaches the ceiling:

| claim | result | where |
|---|---|---|
| 8 real-language facts, out of context, from the store | **100.0%** (warm from a 4-fact base) | r48_bpe8_curr |
| BPE 8 facts, cold | 49.8% | r47_bpe_on |
| store disabled (HC off) | 0.0% | r47_bpe_off |
| scrambled store | 0.0% (gate closes to 0.045) | r47_bpe_shuf |
| facts-in-context ceiling | 96.1% | r47_bpe_ceiling |
| fact count past the char cap | 16f 52.8% / 32f 5.9% cold (curriculum pending) | r48_bpe16/32 |

So the store answers from real-language facts held entirely out of the prompt, at 100% for 8 facts, with matched
controls at zero. The fact count is no longer vocabulary-bound.

## Objective CLOSED on real language — 2026-09-18 (r49-r51)

**Scaling (r49/r50):** the curriculum ladder holds at **100% for 8, 16, and 32 facts** and **99.1% at 64** (warm-
start one rung at a time; 512-slot store does not bind at 64). Replicated on a 2nd seed. Matched scrambled control
**0.16%** (chance) at 64 facts — no leakage at scale.

**Persistence (r51):** a store written by one process, saved, and reloaded in a **fresh process** answers at
**100%** — equal to in-process — while a mismatched store file falls to **0.0%**. The state is a property of the
~16 MB file, not of live interpreter state; the model fingerprint guards against foreign stores.

**Effective context ≫ literal window (r51).** Segments do not cross-attend (`model.forward` is stateless per
segment) — the store is the ONLY channel between them, so the attention window is a fixed `seg_len=128`. Facts are
stated in one window (segment 0); the query is a *separate*, empty 128-token window. The store bridges the gap, and
the reach scales **linearly with store size M**:

| store size M | context bridged | recall from an empty 128-token prompt |
|---|---|---|
| 512  | 3 windows (384 tok)   | 98.8% |
| 1024 | 7 windows (896 tok)   | 98.8% |
| 2048 | 15 windows (1920 tok) | 93.8% |
| 4096 | **31 windows (3968 tok)** | **90.0%** |

reach ≈ M/128 windows; the collapse to 0% lands exactly where `slots_used == M` (the ring buffer laps and clobbers
segment 0's facts). So a 128-token model, with the store, holds and recalls state spanning tens of times its
attention window — a 64-fact set that would itself overflow the window, answered from an empty prompt 31 windows
later.

## ⭐ Store ≫ context window, queried at O(1) — 2026-09-18 (r61)

The flagship demonstration. The model's attention window is `seg_len=128` and segments do not
cross-attend, so the store is the only long-range channel. Fill a store that bridges many windows and
read the query with a **top-k** readout (touch only the k best-scoring slots):

| store size | context bridged | slots | top-1 read | dense read (all slots) |
|---|---|---|---|---|
| M=1024 | 7 windows  | 889  | **100.0%** | 98.8% |
| M=2048 | 15 windows | 1905 | **100.0%** | 93.8% |
| M=4096 | 31 windows (3968 tok) | 3937 | **100.0%** | 90.0% |

**A store 31× the attention window, answered from an empty prompt while each query touches ONE of 3937
slots** — and sparse top-1 *beats* the dense scan (100% vs 90%), because the dense softmax dilutes with
distractor mass that grows with store size while top-1 commits to the exact key match. Effective context
is bounded only by store size, at O(1) slots per query. (The score ranking is still computed densely; a
sparse *candidate pool* — DeepSeek CSA2, route 2 — is how to make it cheap at very large M.)

## CSA2 shared KV store — complete, 2026-09-18 (r64–r65)

DeepSeek-V4.1-Flash (CSA2, arXiv 2609.19969) gets long-context KV cheap by exploiting inter-layer
redundancy. All three of its axes now hold on our store, as an eval-time read path (no retraining):

- **Candidate pool (r64):** restrict the read to index-selected slots — `token` (exact key-token hash)
  scores ~2 of 8001 slots at 63 windows, recall = dense; `ann` (LSH) ~M/12, also = dense. The O(1)-slot
  read (r61/r62) becomes O(1)-**compute**.
- **Cross-layer reuse (r64):** build the pool once at an anchor layer, reuse it at the other read layers
  — identical recall to rebuilding per layer (100% either way). "Few anchor layers, the rest reuse."
- **Low-bit KV (r65):** the store quantizes to **4 bits/element, 8× smaller than fp32, no recall loss**
  (3-bit is a hard cliff, set by the keys; values tolerate 3-bit).

So the store is a **shared KV that holds ≫ the context window (63×), is retrieved in O(1) slots, and
costs 4 bits per element** — the DeepSeek shared-store idea, realised on the in-block memory.

### How to store more (the general recipe)
- **Span / tokens bridged scales LINEARLY with store size M — under a ring buffer.** Widening the store needs **no
  retraining** — store tensors are non-persistent buffers, so an M=512-trained checkpoint runs at M=4096 unchanged.
  Cost is linear store VRAM (`B×M×d_mem`); drop batch as M grows.
- **But the ring buffer is not the ceiling: KEEP THE RIGHT SLOTS (r52).** Effective context under a ring buffer is
  `M/128` windows, but that is a policy limit, not a capacity limit. A fact-protecting eviction (keep the `F=127`
  fact slots, evict oldest prose) bridges **31 windows at 99% from a 256-slot store** — two windows of raw
  capacity. Effective reach is bounded by `F/128` (the facts must fit) plus a working set of recent context, not by
  `M/128`. So "store more" has two axes: grow M, **and** evict by value not recency. (Bonus: a smaller
  value-pruned store reads *cleaner* — less distractor interference — so eviction can raise accuracy, not just save
  space.) **Label-free surprisal-eviction (r53/r54)** approximates the oracle for *distinct* facts: it meets the
  oracle above a threshold `M* ≈ facts + prose-tail` (≈512 slots for a 15-window bridge at 64 facts, ≈1024 for 31
  windows). But it scores *novelty*, not fact-hood, so *repeated* facts (a tiled 16-fact window) go low-surprisal
  and get evicted, capping recall at ~75% regardless of M. Dedup-at-write (r55, keep one copy per token) lifts that
  ceiling back to the oracle's 100% given store headroom (M≥~1024), but not at tight M — where facts, deduped or
  not, still lose the top-M contest to prose first-occurrences. The label-free way to reach the oracle at tight M is
  the WRITE side: don't store the prose at all (`write_select="surprisal"` / a trained binding-aware write gate),
  so the store holds facts + a small working set by construction (open, r56).
- **Distinct facts is curriculum-bound, not M-bound** (r39/r49): more slots do not raise the fact ceiling; the
  warm-start ladder does. M only needs `M ≥ facts × slots-per-fact`.
- **Two prerequisites on real language:** single-token BPE keys (remove the multi-token addressing tax, r36) and
  the curriculum (escape the bootstrapping deadlock, r25).
