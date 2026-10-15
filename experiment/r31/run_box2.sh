#!/usr/bin/env bash
# r31 (box2) — the same comparison at the FULL write rate, where the ring wraps inside a single segment.
set -u
PY=~/research/framework/.venv/bin/python
cd ~/fw
R=experiment/r31
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --n-symbols 128 --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --data data/tinystories.txt
      --dg-bypass --eval-every 250 --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded
      --seg-len 128 --seed 0 --write-threshold 0.6 --key-layer 0 --read-layers 0
      --key-pos-free --freeze-query --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0 --memory-size 192 --n-seg 4 --batch 32 --steps 8000
      --write-select all"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress_box2.log
  $PY -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/${name}_box2.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress_box2.log
}

run r31_ring_m192_full
run r31_ep_m192_full --episode-evict
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress_box2.log
