#!/usr/bin/env bash
# r15 — corpus switch to TinyStories: 512-slot sanity, then the 4x store.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r15
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinystories.txt --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 0
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --write-layer 0 --value-shift 1 --n-steps 0 --beta 32"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r15_ts_M512  --memory-size 512  --n-seg 4  --batch 32 --accum 1 --steps 4000
run r15_ts_M2048 --memory-size 2048 --n-seg 16 --batch 16 --accum 2 --steps 6000
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
