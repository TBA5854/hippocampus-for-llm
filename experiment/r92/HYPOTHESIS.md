# r92 — fine (bigram) key, tested compatibly

## Why
r89 warm-started the fine key from a single-token-key model (r79) → collapsed (confound).
r91 trained collision cold → didn't bootstrap at all (0.2%). Both failed to TEST the idea.
Fix: bootstrap the read path WITH the bigram key from step 0 on the EASY recall task, then
warm the collision from that (compatible warm start).

## Crux + prediction
1. **r92_recall_fine_cold** (recall + `--fine-key`, cold): does the bigram key bootstrap on
   an easy objective? Predict ~high (bigram is a richer but reproducible key; recall has no
   collisions). If it FAILS (~0), the bigram key can't cold-bootstrap → DG-via-bigram is
   dead, write it up and move on.
2. **r92_col_C4_fine_warm** (collision + fine, warm from #1): the fair DG test. Predict
   >> the r89 exact baseline (22% at C4) — ideally near single-fact recall — if the fine
   context term separates colliding facts.
Matched EXACT arm (box2): recall exact cold → collision exact warm (should land ~22% = 1/C),
so fine-vs-exact is a clean same-pipeline comparison.

## Runs
- box1 (fine): r92_recall_fine_cold (12k) → r92_col_C4_fine_warm (6k)
- box2 (exact): r92_recall_exact_cold (12k) → r92_col_C4_exact_warm (6k)
Pass = fine collision >> exact collision (22%) with the recall_fine bootstrap succeeding.
