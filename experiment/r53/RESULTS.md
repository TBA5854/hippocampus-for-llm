# r53 — RESULTS: surprisal is a real label-free signal, but not clean enough to match the oracle at the extremes

Offline `mech.persist evict` with three policies. **box2/main** = 64-fact `r50_bpe64` (seed 1);
**box1/cross** = 16-fact `r50_bpe16_s2` (seed 2). Hypothesis **PARTIALLY confirmed**: surprisal beats
the ring buffer decisively and matches the oracle in the easy regime, but degrades — sometimes to 0% —
at long bridges and tight M, because natural prose carries its own high-surprisal tokens.

## Recall from the empty query window — surprisal vs the ring/oracle bounds

**64 facts (main), M=512:**

| n_seg | bridge | ring | oracle | **surprisal** |
|---|---|---|---|---|
| 4  | 3 win  | 98.8% | 98.8% | **98.8%** |
| 8  | 7 win  | 0.0% | 99.4% | **98.8%** |
| 16 | 15 win | 0.0% | 98.1% | **97.5%** |
| 32 | 31 win | 0.0% | 98.8% | **0.0%** |

**64 facts (main), M=256:** surprisal 100% / 95.6% / 0.0% / 0.0% (holds to 7 win, then prose crowds it out).
**16 facts (cross), M=512:** surprisal 100% / 80.6% / 75.6% / 0.0%. **oracle = 100% at every cell** (16f, both M).

## What it shows

1. **Surprisal works, label-free, in the easy regime.** At 64 facts / M=512 it tracks the oracle within
   ~1 point out to **15 windows (97.5%)** while the ring buffer is at **0%**. So "keep the right slots"
   does **not** require knowing which slots are facts — the write-time `−log p(token)` finds them.

2. **It fails at the extremes, and the failure mode is legible.** Two knobs break it:
   - **Long bridge** (n_seg=32 ⇒ ~3900 prose slots): even M=512 collapses to 0%. With that much prose,
     enough high-surprisal prose tokens (rare words, proper nouns in PG-19) outrank the ~127 fact slots
     and evict them. The oracle, which never scores prose above a fact, does not.
   - **Tight M** (256): the collapse comes sooner (by 15 windows), for the same reason — fewer kept
     slots, so prose wins the competition earlier.

3. **Fewer facts is *harder* for surprisal, not easier.** 16f (75.6% @15win) < 64f (97.5%). The oracle
   is a flat 100% at 16f, so this is not a task-difficulty effect: with 64 facts the seg-0 window is
   packed with 127 distinct high-surprisal fact tokens that dominate the top of the ranking, whereas 16
   tiled facts leave more of the ranking's head contestable by prose. (Confirms r52's oracle across
   fact count and seed as a bonus: 16f seed-2 oracle = 100% throughout.)

## Verdict
- **H53 (surprisal ≈ oracle, label-free):** PARTIALLY CONFIRMED. Surprisal ≫ ring always, and ≈ oracle
  when the store comfortably exceeds the fact-slot count plus a prose working set (64f, M=512, ≤15
  windows). It is NOT falsified into "≈ ring" — it clearly separates facts in the easy regime — but it
  does not reach the oracle at long bridges / tight M, so an unsupervised keep on this signal alone has
  a ceiling below the oracle's.

## Opens — the gap to the oracle
The residual is prose tokens that are high-surprisal by themselves. To close it without labels, the
signal must say "fact" not merely "surprising":
- **surprisal relative to a local prose baseline** (a fact is surprising *given* it should be predicted
  from a stored key, not merely a rare word) — a normalization, still label-free.
- **a trained selective write gate** — r50's gate is near-open (`gates≈1.0`); training it to gate prose
  out (or `write_select="surprisal"` at write time, so prose is never stored) is the learned route.
- Both are the r54 candidates; the oracle's flat 98–100% shows the ceiling they are aiming at.
