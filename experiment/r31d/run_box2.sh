#!/usr/bin/env bash
# r31d (box2) — slot-width compression, and whether it composes with write-rate compression.
set -u
cd ~/fw
PY=~/research/framework/.venv/bin/python
R=experiment/r31d
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --n-symbols 128 --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-model 256 --data data/tinystories.txt
      --dg-bypass --eval-every 250 --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded
      --seg-len 128 --seed 0 --write-threshold 0.6 --key-layer 0 --read-layers 0
      --key-pos-free --freeze-query --query-init key_proj --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0 --memory-size 512 --n-seg 4 --batch 32 --steps 8000
      --no-write-identity"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
  $PY -u -m mech.train $BASE "$@" --out $R/results/$name.json --save $R/ckpt/$name.pt \
      > $R/logs/${name}_box2.log 2>&1
  echo "=== $name EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log
}

run r31d_w64     --d-mem 64
run r31d_w32     --d-mem 32
run r31d_w64_r25 --d-mem 64 --write-select surprisal --write-frac 0.25
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
