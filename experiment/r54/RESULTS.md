# r54 — RESULTS: a clean margin law for surprisal (64f), and the repeat-suppression failure mode (16f)

Offline `mech.persist evict`, policies ring/oracle/surprisal, fixed bridges (15 and 31 windows), fine M
sweep. box2/main = 64-fact `r50_bpe64`; box1/cross = 16-fact `r50_bpe16_s2`. The hypothesis is
**confirmed for 64 facts** (a crisp threshold M\*) and **falsified for 16 facts** in an informative way
— surprisal plateaus below the oracle at any M with eviction on, because tiled fact repeats go
low-surprisal and get evicted.

## 64 facts — surprisal reaches the oracle above a bridge-dependent M\* (recall, surprisal / oracle)

| M | 15-win surprisal | 15-win oracle | 31-win surprisal | 31-win oracle |
|---|---|---|---|---|
| 256  | 0.0%  | 99.4% | 0.0%  | 99.4% |
| 384  | 69.4% | 98.8% | 0.0%  | 98.8% |
| 512  | **97.5%** | 98.1% | 0.0%  | 98.8% |
| 768  | 96.9% | 98.1% | 84.4% | 98.8% |
| 1024 | 95.6% | 97.5% | **95.6%** | 98.1% |
| 1536 | 95.0% | 95.0% | 93.8% | 98.1% |
| 2048 | 93.8% | 93.8% | 93.1% | 94.4% |

**M\* ≈ 512 for a 15-window bridge, ≈ 1024 for a 31-window bridge.** The threshold grows with the
bridge because M\* ≈ (fact slots) + (high-surprisal prose tail at that bridge), and the prose tail grows
with the number of prose segments. Above M\* surprisal is within ~1–3 points of the oracle; the
transition is sharp (0 → 69 → 97 across M=256→384→512 at 15 windows). This is the concrete form of the
r52 recipe: **for a K-window bridge and 64 distinct facts, a surprisal-evicted store of size ≥ M\*(K)
matches the oracle.**

## 16 facts — surprisal PLATEAUS below the oracle: repeat suppression

| M | 15-win surprisal | 31-win surprisal | (oracle, both bridges) |
|---|---|---|---|
| 256  | 2.5%  | 0.0%  | 100% |
| 384  | 73.8% | 0.0%  | 100% |
| 512  | 75.6% | 0.0%  | 100% |
| 768  | 72.5% | 78.8% | 100% |
| 1024 | 71.2% | 76.9% | 100% |
| 1536 | 68.8% | 75.6% | 100% |
| 2048 | 99.4% (no eviction: kept 1905<2048) | 74.4% | 100% |

Surprisal rises to ~72–79% and then **plateaus — more M does not help** — until M exceeds the total
written slots (M=2048, 15-win, kept=1905) where eviction is effectively off and it jumps to the oracle.
So under actual eviction, surprisal never reaches the oracle at 16 facts.

**Mechanism — surprisal marks NOVELTY, not fact-hood.** With 16 facts the seg-0 window is
`k1 v1 … k16 v16` tiled ~4×. Only the *first* occurrence of each fact word is unpredictable; every
tiled repeat is now predicted by the copy the model just saw, so it scores LOW surprisal and is evicted
like prose. About a quarter of the facts lose all their surviving copies to high-surprisal prose,
capping recall at ~75%. With 64 facts the window is filled *once* (64 distinct pairs, no repeats), so
every fact token is first-occurrence high-surprisal and survives — which is exactly why 64f has a clean
M\* and 16f does not.

## Verdict
- **H54 (surprisal reaches oracle above M\*(bridge, n_facts)):** CONFIRMED for 64 distinct facts
  (M\*≈512 @15win, ≈1024 @31win). FALSIFIED for 16 tiled facts — a capacity margin does not fix it;
  surprisal plateaus at ~75% because it scores novelty and repeated facts are not novel.
- Both bridges keep ring at ~0% throughout (its M\* is the full `total`, off-grid) — consistent.

## Opens — r55
The repeat-suppression failure names the fix: score a slot by whether it is a **key→value binding**
(role), not by token novelty. Two routes, both label-free: (a) **dedup at write** so each fact is stored
once and never becomes a low-surprisal repeat (predicts 16f jumps to the 64f-style clean M\*); (b) a
**trained selective write gate** that learns to keep bindings and drop prose (r50's gate is near-open).
Route (a) is a cheap offline test of the mechanism; (b) is the general learned solution.
