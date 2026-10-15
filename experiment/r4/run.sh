#!/usr/bin/env bash
# r4 — close the objective's controls and seeds at 8 facts, then budget vs capacity at 16.
# Sequential (r2: parallel is slower per run on this GPU).
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r4
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --answer-weight 5.0 --batch 32 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinyshakespeare.txt --dg-bypass --eval-every 250
      --lr 0.0003 --memory-size 512 --n-heads 1 --n-layers 6 --n-seg 4 --read-layers 1
      --readout graded --seg-len 128 --write-threshold 0.6"
SAME="--write-layer 5 --key-layer 1 --value-shift 0 --query-init identity"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r4_f8_off   --hc off $SAME --n-facts 8 --steps 4000 --seed 0
run r4_f8_s1    --hc on  $SAME --n-facts 8 --steps 4000 --seed 1
run r4_f8_s2    --hc on  $SAME --n-facts 8 --steps 4000 --seed 2
run r4_f16_12k  --hc on  $SAME --n-facts 16 --steps 12000 --seed 0
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
