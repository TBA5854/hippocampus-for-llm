#!/usr/bin/env bash
# r29 (box1, second pass) — does a FIXED replay budget still hold retention as users accumulate?
# This is the scaling question: with the budget constant, each past user gets a thinner slice.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd ~/research/framework
R=experiment/r29
mkdir -p $R/logs $R/results
RUN=experiment/r23/results/r23_1x_sym128_f8.json
CK=experiment/r23/ckpt/r23_1x_sym128_f8.pt

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.sleep_cycles --run $RUN --ckpt $CK \
      --batch 32 --tag $name "$@" --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r29_cycles12_past --cycles 12 --sleep-steps 600 --sleep-lr 3e-4 --rehearse 1.0 --rehearse-mem 3.0 --replay-past 1.0 --replay-cap 512
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
