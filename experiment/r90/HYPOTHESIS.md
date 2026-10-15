# r90 — curriculum for the write-op gate (fix the gate×depth cold-start)

## Claim
r85: the write-op gate fails to cold-bootstrap in a deep stack (d256/6L, d192/12L at
~0%). r87: the PLAIN store bootstraps those same shapes fine (84.7%, 97.0%). So the
failure is learning the GATE from scratch in a deep stack, not depth. A curriculum should
fix it: warm the gate from a converged plain-store model (the store machinery already
works), then turn the gate on.

## Prediction
Warm `--write-op-gate` from the r87 plain checkpoints (arch2_d256_L6.pt, arch2_d192_L12.pt):
- recall stays HIGH (the read path is already bootstrapped), and
- mem drops from ~508 (plain, all slots) to O(facts) (~200 for 32 facts) as the gate dedups.
If so, the gate×depth cold-start is a curriculum artifact (like the r25/r66 fact-count
cliff), not a capability limit. If recall collapses when the gate turns on even warm, the
gate genuinely conflicts with a deep read path (a deeper finding).

## Runs (box1, warm from r87 plain ckpts, 6k each)
- r90_gate_d256_L6  (warm arch2_d256_L6.pt  + write-op-gate)
- r90_gate_d192_L12 (warm arch2_d192_L12.pt + write-op-gate)
Control = r85 gated cold (0.3% / 0.2%) and r87 plain warm-source (84.7% / 97.0%).
