# r56 — the write-side placement ablation: write @ last block vs write @ every layer

The user asked twice (msg 30, 39): "r start w end vs r/w at every layer, both are gated." r41 settled the
READ half — reading at every layer (88.3%) does not beat a single early read (90.3%), because deep reads
query contextualised states that no longer match the layer-0 (raw-embedding) keys. The WRITE half was
never built: `write_layer` was a single int. This round adds `--write-layers` (the stored value is the
MEAN of the hidden states at the listed layers, same slot count) and runs the A/B.

## Setup — MQAR, the proven objective placement (r11 config, cold 100%)

Everything matched to r11's last-block placement (8 facts, M=512, n_seg 4, key@0, read@0, frozen
identity query, block-0 value skip), varying only where the value is captured:

- **write @ last** — `--write-layer 5` (baseline; r11 = 100% cold).
- **write @ every layer** — `--write-layers 0 1 2 3 4 5` (the ablation).
- **write @ all, scrambled store** — matched control.
- **write @ all, seed 1** — replicate.

    BASE: --task mqar --hc on --no-tile --n-facts 8 ... --read-layers 0 --key-layer 0 --key-pos-free
          --freeze-query --query-init identity --value-skip-layer 0 --memory-size 512 --n-seg 4
    box1: r56_wlast (--write-layer 5) ; r56_wall (--write-layers 0 1 2 3 4 5)
    box2: r56_wall_shuf (--write-layers 0..5 --shuffle-store) ; r56_wall_s1 (--write-layers 0..5 --seed 1)

## Hypothesis

**Claim:** write @ every layer does NOT beat write @ last, and likely underperforms it — averaging early,
content-poor states into the value dilutes the semantically complete last-block representation that r9
showed is the thing worth storing. The key is layer-0 either way, so matching is unaffected; only the
value content changes.

**Predict:** `wlast` ≈ 100% (repro); `wall` at or below it (a tie would say the last block's extra depth
is not load-bearing for the value; a drop confirms the late representation matters). Scrambled at chance.

**Falsified if** `wall` clearly BEATS `wlast` — then multi-depth values carry more than the last block
alone, and the objective's "write at the last block" is leaving recall on the table — a real design fork.

## What it decides
Closes the write-side placement question the read-side (r41) left open, completing the user's
symmetric-placement ask. Either outcome is publishable: a tie/loss confirms the single last-block write
is the right, minimal choice; a win reopens where the value should be captured.
