#!/usr/bin/env bash
# r22 (box2) — 48 facts over a 128-symbol key vocabulary, at the 1x store. Chance is 1/128 = 0.8%.
#
# Why 1x and not 8x: the vocabulary change makes the model a different shape (80 chars + 2x128 symbols
# + 1 = 337 tokens vs 145), so --init cannot warm-start it, and a COLD run at a large store does not
# lift off at all (r17: 4x cold, chance through 12000 steps). Running cold at 8x would test lift-off,
# not vocabulary. 1x trains cold to 100% (r18), so it isolates the key-vocabulary question.
#
# box2 notes: ~/research/framework there belongs to another account and is not writable by us, so work
# lives in ~/fw; uv cannot run either (its cache is owned by that account), so call the venv python directly.
set -u
cd ~/fw
PY=~/research/framework/.venv/bin/python
R=experiment/r22
mkdir -p $R/logs $R/results $R/ckpt

# r18's 1x configuration (plain skip, batch 32) — the one that trains from scratch.
BASE="--task mqar --hc on --no-tile --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinystories.txt --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 0
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0
      --memory-size 512 --n-seg 4 --batch 32 --steps 4000"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
  $PY -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/${name}_box2.log 2>&1
  echo "=== $name EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log
}

run r22_1x_sym128_f48 --n-symbols 128 --n-facts 48
run r22_1x_sym32_f8                                   # matched 32-symbol baseline, same box, same seed
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
