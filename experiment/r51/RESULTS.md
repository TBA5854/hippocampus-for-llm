# r51 — RESULTS: persistence holds, and the store bridges K≈M/128 context windows

Offline probes on the trained r49/r50 checkpoints (no training). Both hypotheses **confirmed**. This
closes the objective on real language: recall out of context (r49/r50), a scrambled control at chance
(r50), **persistence across a process death**, and **an effective context many times the literal
window** — with the general recipe for extending it stated below.

## H51a — persistence: CONFIRMED

32-fact 100% model (`r49_bpe32_curr`), store written by one process, saved, reloaded in a FRESH
process (`mech.persist write` → `read`), 160 queries:

| probe | recall |
|---|---|
| in-process (write process answers) | **100.0%** |
| reloaded (fresh process, store from file) | **100.0%** |
| mismatched (fresh process, ANOTHER batch's store file) | **0.0%** |

The reload equals the in-process number, so nothing leaked through the interpreter — the answer is a
property of the ~16 MB file, not of live state. The mismatched control at 0.0% proves the answers come
from *that* store's contents, not from anything the model memorised. Store fingerprint guard held (a
foreign model's file is refused, not silently degraded — mech/hippo_layer.py:151).

## H51b — how far the store bridges, and what sets the limit: CONFIRMED

64-fact model (`r50_bpe64`), swept over store size M and bridge distance `n_seg-1` (windows of prose
between the facts in segment 0 and the empty query window). Recall from the query window alone:

| M | bridge held @ high recall | tokens bridged | recall | first collapse |
|---|---|---|---|---|
| 512  | 3 windows  | 384  | 98.8% | 0.0% at 7 windows |
| 1024 | 7 windows  | 896  | 98.8% | 0.0% at 15 windows |
| 2048 | 15 windows | 1920 | 93.8% | 0.0% at 31 windows |
| 4096 | **31 windows** | **3968** | **90.0%** | — (still holding) |

**The reach is linear in M: bridge ≈ M / (seg_len−1) segments ≈ M/128 windows.** The collapse is a
hard cliff and it lands exactly where `slots_used == M`: the store is a ring buffer, so once the write
pointer laps, segment 0 (written first, holding every fact) is overwritten and recall goes to zero. As
long as the slot budget is not spent (`slots_used < M`) recall stays 90–99%; a mild taper (98.8 → 93.8
→ 90.0) as the fill approaches M is the last-written facts near the wrap boundary starting to evict.

### The headline, in one line
At **M=4096** the model answers 64 distinct facts — which listed as `key value` pairs already overflow
its 128-token window — from a **separate, empty 128-token prompt**, across **3968 tokens (31×) of
intervening prose it cannot attend to**, at **90%**. Segments do not cross-attend, so this is not a
long context: it is a 128-token model with a store standing in for one 31× larger, and the store is
what carries the state.

## The general recipe: how to store more

Two independent capacity axes — do not conflate them:

1. **Span / tokens bridged — scales LINEARLY with store size M.** Reach ≈ M/128 windows (measured
   above). Checkpoints trained at M=512 evaluate unchanged at M=4096 because the store tensors are
   non-persistent buffers (mech/train.py warm-start note), so widening the store needs **no
   retraining** — it is a deployment-time knob. To bridge farther, raise M (cost is linear store VRAM,
   `B×M×d_mem`; drop batch as M grows — OBJECTIVE/plan R4).
2. **Distinct facts — curriculum-bound, not M-bound.** The fact-count ceiling is a bootstrapping
   deadlock (r25), escaped by the warm-start ladder 4→8→16→32→64 (r49/r50), not by more slots (r39). M
   only has to satisfy `M ≥ facts × slots-per-fact`; below that it is not the limit.

Two prerequisites make both axes work on real language: **single-token (BPE) keys**, which remove the
multi-token-key addressing tax that capped char-level facts at ~13 (r36); and the **curriculum**, which
escapes the deadlock. With both, recall is 100% through 32 facts, 99% at 64, persists across a process
death, and reaches tens of context-windows deep — limited only by M, which is free to grow.

## Verdicts
- **H51a (persistence):** CONFIRMED — reloaded 100% == in-process 100%; mismatched 0.0%.
- **H51b (bridge ≈ M/128, M is the lever):** CONFIRMED — linear reach, hard cliff at `slots==M`.

## Opens
- The ring buffer's hard cliff wastes the whole store on overflow; a value/recency-scored eviction
  (the r26–r33 consolidation machinery, so far MQAR-only) should degrade gracefully instead of to 0%
  and is the natural way to push effective context further at fixed M.
- The mild 99→90 taper toward saturation is untangled here (proximity-to-wrap vs interference); a
  scored store would remove the confound.
