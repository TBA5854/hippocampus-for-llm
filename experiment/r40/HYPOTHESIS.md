# r40 — extend the language curriculum, compose it with shift-3, replicate it

r38 confirmed the curriculum escapes the language fact-cliff: 8 facts warm from the 4-fact checkpoint reached 46.4%
(pos0 11.4% -> 55.9%), where cold was 10.3%. This round pushes on three fronts.

- **H40a — the ladder continues.** 12 facts warm from the 8-fact curriculum checkpoint (`r38_curr8`) should beat
  12-facts-cold (r37_f12: 6.6%), mirroring r25's 16->48 ladder on MQAR. Falsified if it lands at cold's ~7%.
- **H40b — shift-3 and curriculum compose (H38c).** 8-fact shift-3 warm from the 4-fact shift-3 checkpoint
  (`r38_f4_s3`). Shift-3 addresses off the delimiter (r37: cold 8-fact 31.6%); the curriculum escapes bootstrapping
  (r38: cold 8-fact -> warm 46.4%). If they attack different bottlenecks the warm shift-3 should beat both. Falsified
  if it does not exceed 46.4%.
- **H40c — the curriculum replicates.** Seed 2 of the 8-fact warm start. r38_curr8 was one seed; r34 warned this
  architecture can swing 3x across seeds. Falsified if seed 2 lands near cold's 10%.

BASE = r38's, seed 1 (H40c uses seed 2). Warm starts via `--init` on r38 checkpoints (store buffers are
non-persistent, so a checkpoint trained at one fact count loads into another — the r19/r25 mechanism).
