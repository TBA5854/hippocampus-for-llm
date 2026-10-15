#!/usr/bin/env bash
# r1 — write at block 4 (repro) vs block 5 (objective), plus both controls.
# Base flags are st_bg4's exact args, recovered from runs/st_bg4.json.
# Runs sequentially: the A4000 is shared, and four at once would contend.
set -u
# A detached setsid shell is non-login, so ~/.local/bin (where uv lives) is not on PATH.
# The first launch exited 127 on every run for exactly this reason.
export PATH="$HOME/.local/bin:$PATH"
cd ~/research/framework
mkdir -p experiment/r1/logs experiment/r1/results

BASE="--task mqar --n-facts 2 --answer-weight 5.0 --batch 32 --beta-grad 4.0 --beta-max 64.0
      --d-mem 128 --d-model 256 --data data/tinyshakespeare.txt --dg-bypass --eval-every 250
      --lr 0.0003 --memory-size 512 --n-heads 1 --n-layers 6 --n-seg 4 --read-layers 1
      --readout graded --seed 0 --seg-len 128 --steps 4000 --write-threshold 0.6"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" | tee -a experiment/r1/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out experiment/r1/results/$name.json > experiment/r1/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" | tee -a experiment/r1/logs/_progress.log
}

run r1_w4      --hc on --write-layer 4
run r1_w5      --hc on --write-layer 5
run r1_w5_shuf --hc on --write-layer 5 --shuffle-store
run r1_w5_off  --hc off --write-layer 5
echo "=== ROUND DONE $(date -Is)" | tee -a experiment/r1/logs/_progress.log
