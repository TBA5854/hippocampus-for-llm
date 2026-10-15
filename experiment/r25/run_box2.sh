#!/usr/bin/env bash
# r25 (box2) — a curriculum over FACT COUNT (not store size): 8 -> 16 -> 48 facts at 128 symbols, 1x store.
# Each run warm-starts from the previous one's checkpoint; the vocabulary is fixed at 128 throughout so the
# model shapes match (a vocabulary change would make --init impossible, see r22).
set -u
cd ~/fw
PY=~/research/framework/.venv/bin/python
R=experiment/r25
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinystories.txt --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 0
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0
      --memory-size 512 --n-seg 4 --batch 32 --steps 4000 --n-symbols 128"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
  $PY -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/${name}_box2.log 2>&1
  echo "=== $name EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log
}

run r25_1x_f16_warm      --n-facts 16 --init experiment/r23/ckpt/r23_1x_sym128_f8.pt
run r25_1x_f48_warm      --n-facts 48 --init experiment/r25/ckpt/r25_1x_f16_warm.pt
run r25_1x_f48_warm_shuf --n-facts 48 --init experiment/r25/ckpt/r25_1x_f16_warm.pt --shuffle-store
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
