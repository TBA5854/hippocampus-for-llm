#!/usr/bin/env bash
# r51 — close the objective on real language: persistence + context-window sweep.
# OFFLINE probes on trained checkpoints (no training). Runs on box2 (~/fw).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r51
mkdir -p $R/logs $R/results $R/store || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
echo "r51 starting in $ROOT at $(date -Is)" | tee $R/logs/_progress.log
UV="uv run --group sleep python -u -m mech.persist"

R49=experiment/r49/results/r49_bpe32_curr.json
C49=experiment/r49/ckpt/r49_bpe32_curr.pt
R50=experiment/r50/results/r50_bpe64.json
C50=experiment/r50/ckpt/r50_bpe64.pt

{
  echo "=== H51a persistence (32-fact 100% model) START $(date -Is)"
  $UV write --run $R49 --ckpt $C49 --dir $R/store --iters 10 --batch 16
  $UV read  --run $R49 --ckpt $C49 --dir $R/store --iters 10 --batch 16
  $UV read  --run $R49 --ckpt $C49 --dir $R/store --iters 10 --batch 16 --mismatch
  echo "=== H51b context sweep (64-fact model) START $(date -Is)"
  $UV context --run $R50 --ckpt $C50 --iters 10 --batch 16 \
      --mem-list 512 1024 2048 4096 --nseg-list 4 8 16 32 \
      --out $R/results/r51_context.json
  echo "=== ROUND DONE $(date -Is)"
} > $R/logs/r51.log 2>&1
tail -1 $R/logs/r51.log
