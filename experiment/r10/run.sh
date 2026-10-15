#!/usr/bin/env bash
# r10 — single-occurrence facts: skip path for the last-block write; lock in the block-0 observation store.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r10
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --answer-weight 5.0 --batch 32 --beta-grad 4.0
      --beta-max 64.0 --d-mem 256 --d-model 256 --data data/tinyshakespeare.txt --dg-bypass
      --eval-every 250 --lr 0.0003 --memory-size 512 --n-heads 1 --n-layers 6 --n-seg 4
      --readout graded --seg-len 128 --steps 4000 --write-threshold 0.6
      --key-layer 0 --read-layers 0 --key-pos-free --freeze-query --query-init identity
      --value-shift 1"
OBS="--write-layer 0"                     # r9_w0s1: 100% on single occurrences

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r10_w5skip0   --write-layer 5 --value-skip-layer 0 --n-facts 8  --seed 0
run r10_obs_s1    $OBS --n-facts 8  --seed 1
run r10_obs_shuf  $OBS --n-facts 8  --seed 0 --shuffle-store
run r10_obs_f16   $OBS --n-facts 16 --seed 0
run r10_obs_f32   $OBS --n-facts 32 --seed 0
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
