#!/usr/bin/env bash
# r22 (box1) — 16x store (M=8192, 64 segments) warm-started from the 8x model.
# Per-example cost is 4x r20's, so: batch 2 x accum 16 (effective 32), 500 steps, eval every 125.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r22
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinystories.txt --dg-bypass --eval-every 125
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 0
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0 --value-skip-norm
      --memory-size 8192 --n-seg 64 --batch 2 --accum 16 --steps 500
      --init experiment/r20/ckpt/r20_warm_skipnorm_M4096.pt"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r22_16x
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
