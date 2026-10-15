#!/usr/bin/env bash
# r28 (box2) — does rehearsing the MEMORY task keep the write path calibrated through sleep?
set -u
cd ~/fw
PY=~/research/framework/.venv/bin/python
R=experiment/r28
mkdir -p $R/logs $R/results

RUN=experiment/r23/results/r23_1x_sym128_f8.json
CK=experiment/r23/ckpt/r23_1x_sym128_f8.pt

run() {
  name=$1; shift
  echo "=== $name START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
  $PY -u -m mech.sleep_evict --run $RUN --ckpt $CK --batch 64 --shared-facts --tag $name "$@" \
      --out $R/results/$name.json > $R/logs/${name}_box2.log 2>&1
  echo "=== $name EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log
}

run r28_mem0        --sleep-steps 1500 --sleep-lr 3e-4 --rehearse 1.0
run r28_mem1        --sleep-steps 1500 --sleep-lr 3e-4 --rehearse 1.0 --rehearse-mem 1.0
run r28_mem3        --sleep-steps 1500 --sleep-lr 3e-4 --rehearse 1.0 --rehearse-mem 3.0
run r28_mem1_s4000  --sleep-steps 4000 --sleep-lr 3e-4 --rehearse 1.0 --rehearse-mem 1.0
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
