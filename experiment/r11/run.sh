#!/usr/bin/env bash
# r11 — lock in the last-block write + block-0 skip on single-occurrence facts: budget, seed, control, scale.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r11
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --answer-weight 5.0 --batch 32 --beta-grad 4.0
      --beta-max 64.0 --d-mem 256 --d-model 256 --data data/tinyshakespeare.txt --dg-bypass
      --eval-every 250 --lr 0.0003 --memory-size 512 --n-heads 1 --n-layers 6 --n-seg 4
      --readout graded --seg-len 128 --steps 8000 --write-threshold 0.6
      --key-layer 0 --read-layers 0 --key-pos-free --freeze-query --query-init identity
      --value-shift 1 --write-layer 5 --value-skip-layer 0"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r11_skip_s0    --n-facts 8  --seed 0
run r11_skip_s1    --n-facts 8  --seed 1
run r11_skip_shuf  --n-facts 8  --seed 0 --shuffle-store
run r11_skip_f32   --n-facts 32 --seed 0
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
