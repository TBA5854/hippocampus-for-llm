#!/usr/bin/env bash
# r29 (box2, second pass) — replaying the past under a FIXED nightly budget.
# The first attempt replayed every user in full and OOMed by ~user 5; --replay-cap bounds it,
# which is also the realistic design (a night is finite).
set -u
cd ~/fw
PY=~/research/framework/.venv/bin/python
R=experiment/r29
mkdir -p $R/logs $R/results
RUN=experiment/r23/results/r23_1x_sym128_f8.json
CK=experiment/r23/ckpt/r23_1x_sym128_f8.pt

run() {
  name=$1; shift
  echo "=== $name START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
  $PY -u -m mech.sleep_cycles --run $RUN --ckpt $CK --batch 32 --tag $name "$@" \
      --out $R/results/$name.json > $R/logs/${name}_box2.log 2>&1
  echo "=== $name EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log
}

run r29_cycles6_past --cycles 6 --sleep-steps 600 --sleep-lr 3e-4 --rehearse 1.0 --rehearse-mem 3.0 --replay-past 1.0 --replay-cap 512
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
