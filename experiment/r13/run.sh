#!/usr/bin/env bash
# r13 — separate store size from batch size with gradient accumulation (effective batch 32).
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r13
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinyshakespeare.txt --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --steps 6000 --seed 0
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --write-layer 0 --value-shift 1"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

# 1. does 8 x 4 accumulation reproduce the known 512-slot result (r9_w0s1: 100%, batch 32)?
run r13_M512_b8a4   --memory-size 512  --n-seg 4 --batch 8  --accum 4
# 2. 2x store at effective batch 32 (r12_M1024 ran it at batch 16, no accumulation)
run r13_M1024_b16a2 --memory-size 1024 --n-seg 8 --batch 16 --accum 2
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
