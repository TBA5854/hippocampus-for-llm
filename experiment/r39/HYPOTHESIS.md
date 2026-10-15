# r39 — is the language fact-cliff capacity? and does a deduped store read near zero?

Two questions the user raised, both answerable on box2 while r38 runs.

## Q1 — store size vs fact count. Prediction: store size is NOT the lever.

r37 found 4 facts -> 90.3%, 8 facts -> 10.3% at a fixed **512-slot** store. Facts live in segment 0, tiled, and
~381 slots get written per example (3 segments x 127), so at M=512 the store is 74% full and **already holds all 8
facts** — the cliff is not "no room". MQAR settled this from the other direction: r24 put the same cliff at 8->16,
r25 proved it is the **bootstrapping deadlock, not capacity** (curriculum -> 48 facts 100%), and r13 showed store
size only *delays lift-off*, never caps recall.

So doubling / quadrupling the store at fixed 8 facts adds only **empty headroom**. If capacity were the limit,
recall would rise with M; the diagnosis predicts it stays flat at ~10%.

- **H39a — flat.** `m1024_f8` and `m2048_f8` land near `m512_f8`'s 10.3%. Falsified if recall rises with M, which
  would mean the language cliff IS capacity where MQAR's was not — itself a finding.
- The lever that DOES work (curriculum) is r38, running in parallel.

## Q2 — deduped store: is the read near zero?

The store is tiled, so it holds many near-duplicate slots per fact. "Dedup" = keep far fewer slots. Two existing
results bracket it: single-occurrence facts (the deduped *input*) reached **100%** on MQAR (r9, r11), and
content-aware slot reduction (surprisal gate) was **free to 4x** (r30, 100%). So the prior is **not near zero** —
redundancy is not what the read depends on.

The one place a "dedup" read IS near chance is r36's delimiter: the `:` key has one near-duplicate slot per fact,
and the read over those duplicates is exactly **1/n_facts** — near zero for large n. That is duplicate *keys*, not a
deduped store.

Test it directly on the language task at 4 facts (which works at 90.3%), squeezing the slot count with the
surprisal gate:

- **H39b — aggressive dedup holds.** `f4_frac25` (keep 25%) and `f4_frac10` (keep 10%) stay well above chance. If
  they collapse toward 0, the tiling redundancy is load-bearing and dedup DOES kill the read — the opposite of the
  MQAR prior, and worth knowing before any compression is trusted on language.

## Runs (box2, seed 1, ~12 min each)

```
BASE = r37/r38's LANG_BASE (r23's), seed 1, 4 segments, seg_len 128

r39_m1024_f8   --n-facts 8  --memory-size 1024      Q1
r39_m2048_f8   --n-facts 8  --memory-size 2048      Q1
r39_f4_frac25  --n-facts 4  --write-select surprisal --write-frac 0.25   Q2
r39_f4_frac10  --n-facts 4  --write-select surprisal --write-frac 0.10   Q2
```

Baselines already in hand: m512_f4 90.3%, m512_f8 10.3%, m512_f12 6.6% (r37/r36).

Width (d_mem) is deliberately NOT swept: r31d already settled it — narrowing the slot to 64/32 drops to chance
(slot@1 0.0%), because a projection destroys the exact-token match. Width is pinned at d_model=256; it is a hard
constraint, not a free axis.
