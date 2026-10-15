#!/usr/bin/env bash
# r20 control on box 2 (<user>@<box2-host>) — the matched scrambled-store run, in parallel with
# box 1's real run. Same BASE as run.sh; only --shuffle-store differs.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r20
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinystories.txt --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 0
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0 --value-skip-norm
      --memory-size 4096 --n-seg 32 --batch 4 --accum 8 --steps 1500
      --init experiment/r19/ckpt/r19_warm_skipnorm_M2048.pt"

name=r20_warm_skipnorm_M4096_shuf
echo "=== $name START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
uv run --group sleep python -u -m mech.train $BASE --shuffle-store \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/${name}_box2.log 2>&1
echo "=== $name EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
