#!/usr/bin/env bash
# r19 — warm-start the 4x last-block write from the 2x normalised-skip model; matched scrambled control.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r19
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinystories.txt --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 0
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0 --value-skip-norm
      --memory-size 2048 --n-seg 16 --batch 16 --accum 2 --steps 4000
      --init experiment/r18/ckpt/r18_skipnorm_M1024.pt"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r19_warm_skipnorm_M2048
run r19_warm_skipnorm_M2048_shuf --shuffle-store
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
