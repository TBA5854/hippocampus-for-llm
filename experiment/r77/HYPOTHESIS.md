# r77 — settle r76: does the bounded gate reach full recall (given convergence)?

r76 left the recall arm inconclusive: gate_recall was still climbing (66.1% and
rising) when trimmed at 3k, vs no-gate 94.1%. r77 runs the SAME crude exact-key gate
to convergence (6k) to separate under-training from a real cap.

Warm from r70_pl_recall (per-layer base), per-layer store, `--write-op-gate`, M=512,
f16, seed 0. Report recall AND mem occupancy at convergence.
- box1: r77_gate_recall_6k (bpe_recall).
- box2: r77_gate_update_6k (bpe_update).

Baselines (r76, final): recall no-gate 94.1% (mem 508); update gate 51.2% / no-gate
7.8%.

Verdicts:
- gate_recall -> ~90%+ while mem stays bounded (~O(facts)) => the managed memory is
  FULLY VIABLE (bounded AND recall-preserving) => greenlight one store + adapters (r79),
  and multi-hop-as-working-memory (r78) is on solid ground.
- gate_recall plateaus clearly below no-gate while still bounded => the crude
  dedup-by-token rule caps recall (prose-clobber: a fact's key token recurs in prose
  and overwrites its slot). Then r78 = salience/learned gate (don't overwrite a fact
  with filler) BEFORE the one-store step.
Either outcome is decisive for the checkpoint. Report best AND final + mem.
