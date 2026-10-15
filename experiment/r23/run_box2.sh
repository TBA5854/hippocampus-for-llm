#!/usr/bin/env bash
# r23 (box2) — separate key vocabulary from fact count, at the 1x store, cold, with r18's full budget.
# box2 notes: ~/research/framework there belongs to another account and is not writable by us, so work
# lives in ~/fw; uv cannot run either (its cache is owned by that account), so call the venv python directly.
set -u
cd ~/fw
PY=~/research/framework/.venv/bin/python
R=experiment/r23
mkdir -p $R/logs $R/results $R/ckpt

# r18's 1x configuration (plain skip, batch 32) — the one that trains from scratch — at its full 8000 steps.
BASE="--task mqar --hc on --no-tile --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinystories.txt --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 0
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0
      --memory-size 512 --n-seg 4 --batch 32 --steps 8000"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
  $PY -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/${name}_box2.log 2>&1
  echo "=== $name EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log
}

run r23_1x_sym128_f8  --n-symbols 128 --n-facts 8      # vocabulary alone
run r23_1x_sym32_f8   --n-symbols 32  --n-facts 8      # reference line (r18 on box2)
run r23_1x_sym128_f48 --n-symbols 128 --n-facts 48     # r22's failure, full budget
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
