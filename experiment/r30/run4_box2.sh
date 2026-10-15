#!/usr/bin/env bash
# r30 (box2) — matched scrambled control for the compressed store.
# Every other round has one; the compression result needs the same discipline. Same warm start,
# same rate, same steps — only --shuffle-store differs. (Checkpoint copied from box1.)
set -u
cd ~/fw
PY=~/research/framework/.venv/bin/python
R=experiment/r30
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --n-symbols 128 --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --data data/tinystories.txt
      --dg-bypass --eval-every 250 --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded
      --seg-len 128 --seed 0 --write-threshold 0.6 --key-layer 0 --read-layers 0
      --key-pos-free --freeze-query --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0 --memory-size 512 --n-seg 4 --batch 32 --steps 4000"

echo "=== r30_surp12_warm_shuf START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
$PY -u -m mech.train $BASE --write-select surprisal --write-frac 0.125 --shuffle-store \
    --init experiment/r30/ckpt/r30_surp25.pt \
    --out $R/results/r30_surp12_warm_shuf.json > $R/logs/r30_surp12_warm_shuf_box2.log 2>&1
echo "=== r30_surp12_warm_shuf EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
