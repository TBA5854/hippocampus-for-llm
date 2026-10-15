#!/usr/bin/env bash
# r27 (box2) — does sleep actually reclaim slots, and at what cost?
# Sweep: protection (none / keys frozen + rehearsal) x sleep budget, on the 8-fact checkpoint
# (fewer facts to consolidate than the 48-fact one, so it is the easiest case — if it fails here
# the loop does not work at all).
set -u
cd ~/fw
PY=~/research/framework/.venv/bin/python
R=experiment/r27
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

run r27_naive_s400            --sleep-steps 400  --sleep-lr 1e-4
run r27_prot_s400             --sleep-steps 400  --sleep-lr 1e-4 --freeze-keys --rehearse 1.0
run r27_prot_s1500            --sleep-steps 1500 --sleep-lr 1e-4 --freeze-keys --rehearse 1.0
run r27_prot_s1500_lr3e4      --sleep-steps 1500 --sleep-lr 3e-4 --freeze-keys --rehearse 1.0
run r27_prot_s4000_lr1e3      --sleep-steps 4000 --sleep-lr 1e-3 --freeze-keys --rehearse 1.0
run r27_nofreeze_s1500        --sleep-steps 1500 --sleep-lr 3e-4 --rehearse 1.0
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
