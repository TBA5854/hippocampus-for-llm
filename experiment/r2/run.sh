#!/usr/bin/env bash
# r2 — align the store to the last block: shift 0, and early keys. Parallel runs.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r2
mkdir -p $R/logs $R/results

BASE="--task mqar --hc on --n-facts 2 --answer-weight 5.0 --batch 32 --beta-grad 4.0
      --beta-max 64.0 --d-mem 128 --d-model 256 --data data/tinyshakespeare.txt --dg-bypass
      --eval-every 250 --lr 0.0003 --memory-size 512 --n-heads 1 --n-layers 6 --n-seg 4
      --read-layers 1 --readout graded --seed 0 --seg-len 128 --steps 4000 --write-threshold 0.6"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r2_w4          --write-layer 4 --value-shift 1 &
run r2_w5s1        --write-layer 5 --value-shift 1 &
run r2_w5s0        --write-layer 5 --value-shift 0 &
run r2_w5s0k1      --write-layer 5 --value-shift 0 --key-layer 1 &
run r2_w5s0k1_shuf --write-layer 5 --value-shift 0 --key-layer 1 --shuffle-store &
wait
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
