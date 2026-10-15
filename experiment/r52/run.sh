#!/usr/bin/env bash
# r52 — fact-protecting eviction vs ring buffer. OFFLINE probe on r50_bpe64. box2 (~/fw).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r52
mkdir -p $R/logs $R/results || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
echo "r52 starting in $ROOT at $(date -Is)" | tee $R/logs/_progress_box2.log
UV="uv run --group sleep python -u -m mech.persist"

R50=experiment/r50/results/r50_bpe64.json
C50=experiment/r50/ckpt/r50_bpe64.pt

{
  echo "=== r52 evict (ring vs oracle) START $(date -Is)"
  $UV evict --run $R50 --ckpt $C50 --iters 10 --batch 16 \
      --mem-list 256 512 --nseg-list 4 8 16 32 \
      --out $R/results/r52_evict.json
  echo "=== ROUND DONE $(date -Is)"
} > $R/logs/r52_box2.log 2>&1
tail -1 $R/logs/r52_box2.log
