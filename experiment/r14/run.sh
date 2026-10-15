#!/usr/bin/env bash
# r14 — one-shot read (no settling) so a 4x store is affordable; validate at 512, then run 4x.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r14
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinyshakespeare.txt --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 0
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --write-layer 0 --value-shift 1
      --n-steps 0 --beta 32"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

# 1. does a one-shot read reproduce the 512-slot result (r9_w0s1 / r13_M512_b8a4: 100%)?
run r14_M512_oneshot  --memory-size 512  --n-seg 4  --batch 32 --accum 1 --steps 4000
# 2. 4x store at effective batch 32, one-shot read
run r14_M2048_oneshot --memory-size 2048 --n-seg 16 --batch 8  --accum 4 --steps 6000
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
