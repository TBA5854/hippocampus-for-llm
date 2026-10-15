#!/usr/bin/env bash
# r24 (box2) — where between 8 and 48 facts does the 1x store break? Vocabulary held at 128 throughout,
# so this moves ONE variable (r22's mistake was moving both).
set -u
cd ~/fw
PY=~/research/framework/.venv/bin/python
R=experiment/r24
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinystories.txt --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 0
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0
      --memory-size 512 --n-seg 4 --batch 32 --steps 8000 --n-symbols 128"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
  $PY -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/${name}_box2.log 2>&1
  echo "=== $name EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log
}

run r24_1x_sym128_f16 --n-facts 16
run r24_1x_sym128_f32 --n-facts 32
run r24_1x_sym128_f24 --n-facts 24
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
