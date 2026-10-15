#!/usr/bin/env bash
# r6 — scale the layer-0 store: replicate, control, facts 16/32, store size x4.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r6
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --answer-weight 5.0 --batch 32 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinyshakespeare.txt --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --steps 4000
      --write-threshold 0.6 --seed 0
      --write-layer 5 --key-layer 0 --read-layers 0 --value-shift 0 --query-init identity"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r6_f8_s2     --n-facts 8  --memory-size 512  --n-seg 4 --seed 2
run r6_f8_shuf   --n-facts 8  --memory-size 512  --n-seg 4 --shuffle-store
run r6_f16       --n-facts 16 --memory-size 512  --n-seg 4
run r6_f32       --n-facts 32 --memory-size 512  --n-seg 4
run r6_f8_M2048  --n-facts 8  --memory-size 2048 --n-seg 16
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
