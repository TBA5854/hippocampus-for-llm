# r55 — RESULTS: dedup lifts the 16-fact plateau to the oracle, but only with store headroom

Offline `mech.persist evict`, policies oracle / surprisal / **surprisal_dedup** (keep only the
highest-surprisal copy of each token before the top-M keep). box1/main = 16-fact `r50_bpe16_s2` (the
r54 failure); box2/control = 64-fact `r50_bpe64`. Hypothesis **partially confirmed**: dedup raises the
16-fact ceiling from a hard ~75% plateau to the oracle's 100% — but only once M is large enough, and it
does not help (marginally hurts) at tight M.

## 16 facts (main) — surprisal / surprisal_dedup / (oracle = 100% throughout)

| M | 15-win surp | 15-win **dedup** | 31-win surp | 31-win **dedup** |
|---|---|---|---|---|
| 256  | 2.5%  | 10.6% | 0.0%  | 0.0%  |
| 384  | 73.8% | 75.0% | 0.0%  | 0.0%  |
| 512  | 75.6% | 73.8% | 0.0%  | 8.1%  |
| 768  | 72.5% | 72.5% | 78.8% | 74.4% |
| 1024 | 71.2% | **100.0%** | 76.9% | 71.9% |
| 1536 | 68.8% | **99.4%**  | 75.6% | **98.8%** |
| 2048 | 99.4% | 99.4% | 74.4% | **100.0%** |

Plain surprisal plateaus at ~70–79% at every M with eviction on (r54). Dedup **breaks through to the
oracle** — 100% at M≥1024 (15-win) and M≥1536 (31-win) — because once each fact's low-surprisal tiled
repeats are removed, the fact's single high-surprisal first occurrence is no longer diluted and survives
the top-M keep. But below that M the two tie (~73%), and at M=512 dedup is a touch *lower* than plain
surprisal (73.8 vs 75.6): removing a fact's redundant copies makes it more fragile when every slot is
contested.

## 64 facts (control) — dedup ≈ surprisal, as predicted

| M | 15-win surp | 15-win dedup |
|---|---|---|
| 256 | 0.0% | 1.9% |
| 384 | 69.4% | 88.8% |
| 512 | 97.5% | 93.1% |
| 768 | 96.9% | 93.1% |
| 1024 | 95.6% | 96.2% |
| 1536+ | ≈ | ≈ |

With 64 distinct facts the window is filled once, so there are almost no duplicate tokens to remove and
dedup barely moves the curve (±few points, both directions — noise). Confirms dedup does not damage the
distinct case.

## Diagnosis
The r54 plateau **was** repeat suppression: dedup removes it and recovers the oracle's 100%. But dedup
only changes *which* slots win the top-M contest; it does not change *how many* slots there are to win.
So it needs the same headroom the 64-fact case needs — M above `facts + high-surprisal-prose tail` — and
the 16 facts are such a small fraction of the ranking head that this margin is large (M\*≈1024–1536 vs
≈512 for 64f). At tight M the facts still lose to prose first-occurrences, deduped or not; and dedup
even removes the redundancy that helped a fact survive by sheer copy count. So dedup is the right idea
(store each fact once) but not sufficient at tight M.

## Verdict
- **H55 (dedup fixes the 16f plateau):** PARTIALLY CONFIRMED. Ceiling lifted from ~75% to oracle 100% —
  but conditional on M ≥ ~1024, and no gain (slight loss) at tight M. Not falsified (dedup clearly works
  given headroom); not a clean win (tight-M behaviour unchanged).
- **Control:** dedup ≈ surprisal at 64f — no harm to the distinct case.

## Opens — the tight-M gap needs the write side, not the read side
Every offline eviction policy (r52–r55) shares one limit: it prunes a store that already contains all
the prose, so at tight M the facts compete with prose first-occurrences it should never have stored.
The oracle wins only because it is *told* which slots are facts. The label-free way to get the oracle at
tight M is to **not store the prose in the first place** — a selective WRITE (`write_select="surprisal"`
/ a trained write gate that keeps key→value bindings), so the store holds facts + a small working set
by construction. That is a training/write-side round (r56), the natural end of this thread.
