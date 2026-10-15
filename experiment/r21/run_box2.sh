#!/usr/bin/env bash
# r21 (box2) — 32 facts at the 8x store.
# box2 note: ~/research/framework there is owned by another account (adminuser) and is NOT writable by us —
# checkpoint/result writes fail with "Permission denied" even though log writes appeared to work at first.
# So box2 work lives in ~/fw (ours), and we call adminuser's venv python directly: uv itself cannot run
# because ~/.cache/uv is adminuser-owned too.
set -u
cd ~/fw
PY=~/research/framework/.venv/bin/python
R=experiment/r21
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinystories.txt --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 0
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0 --value-skip-norm
      --memory-size 4096 --n-seg 32 --batch 4 --accum 8 --steps 1500
      --init experiment/r19/ckpt/r19_warm_skipnorm_M2048.pt"

name=r21_8x_f32
echo "=== $name START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
$PY -u -m mech.train $BASE --n-facts 32 \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/${name}_box2.log 2>&1
echo "=== $name EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
