# r69 — diagnose the r68 multi-hop negative

r68: 2-hop chaining failed at chance, and the models never fit the task. Before
concluding "a raw retrieved value can't be an exact key", rule out the cheaper
explanations. Three checks (see r68/RESULTS):

box2 (root ~/fw, warm r66_mlcold_f16):
- **hop1_retrieved** — `n_hops 1` (query A -> target B) = plain recall in the
  pad-separated triple format. Is the hop-task FORMAT learnable at all? If this
  fails, the format/warm-start is the problem, not chaining.
- **hop2_curr** — `n_hops 2`, warm from hop1_retrieved. Curriculum: does
  establishing 1-hop first unlock 2-hop (as curriculum unlocked fact count, r49/r66)?

box1 (root ~/research/framework, warm r66_mlcurr_f16):
- **hop1_token** — `n_hops 1 --hop-key token`: sanity that single-hop recall works
  in this format independent of the chaining path.
- **hop2_long** — `n_hops 2 --hop-key retrieved`, 12k steps: does long training
  alone crack 2-hop (as it cracked the cold cliff in r66)?

Verdicts:
- hop1 works, hop2 (curr AND long) stays ~0% -> the retrieved value cannot serve as
  an exact key; multi-hop needs a re-projection into key space or the thalamic
  relay (architectural, not a knob).
- hop1 ALSO fails -> the hop task / warm-start is broken; fix the task before any
  chaining claim.
- hop2_curr works -> curriculum unlocks chaining; multi-hop is a training-schedule
  problem, and the shared store CAN chain.
Report best AND final.
