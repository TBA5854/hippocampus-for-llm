#!/usr/bin/env bash
# r32 (box2) — seed 2 of both configurations.
# Seed 0: full-write 99.7%, 25% surprisal 100.0%. Seed 1: full-write 27.8%, 25% surprisal 100.0%.
# A third seed decides whether "the redundant store is the fragile one" is a real property or one bad draw.
set -u
cd ~/fw
PY=~/research/framework/.venv/bin/python
R=experiment/r32
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --n-symbols 128 --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --data data/tinystories.txt
      --dg-bypass --eval-every 250 --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded
      --seg-len 128 --seed 2 --write-threshold 0.6 --key-layer 0 --read-layers 0
      --key-pos-free --freeze-query --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0 --memory-size 512 --n-seg 4 --batch 32 --steps 8000"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
  $PY -u -m mech.train $BASE "$@" --out $R/results/$name.json --save $R/ckpt/$name.pt \
      > $R/logs/${name}_box2.log 2>&1
  echo "=== $name EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log
}

run r32_surp25_s2 --write-select surprisal --write-frac 0.25
run r32_full_s2   --write-select all
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
