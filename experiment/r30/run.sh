#!/usr/bin/env bash
# r30 (box1) — surprisal arm: write only where the token a slot returns was unpredictable.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd ~/research/framework
R=experiment/r30
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --n-symbols 128 --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --data data/tinystories.txt
      --dg-bypass --eval-every 250 --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded
      --seg-len 128 --seed 0 --write-threshold 0.6 --key-layer 0 --read-layers 0
      --key-pos-free --freeze-query --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0 --memory-size 512 --n-seg 4 --batch 32 --steps 8000"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r30_surp50 --write-select surprisal --write-frac 0.5
run r30_surp25 --write-select surprisal --write-frac 0.25
run r30_surp12 --write-select surprisal --write-frac 0.125
run r30_last   --write-select last
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
