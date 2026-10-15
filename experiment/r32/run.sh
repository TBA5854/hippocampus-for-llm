#!/usr/bin/env bash
# r32 (box1) — does the sleep/eviction loop still work on a COMPRESSED store?
# r30 measured fact coverage (100% at the 25% rate) and argued replay should survive, but the two
# halves of the system have never been run together. This is that test, on r30's 25% checkpoint.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd ~/research/framework
R=experiment/r32
mkdir -p $R/logs $R/results

RUN=experiment/r30/results/r30_surp25.json
CK=experiment/r30/ckpt/r30_surp25.pt

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.sleep_evict --run $RUN --ckpt $CK \
      --batch 64 --shared-facts --tag $name "$@" \
      --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r32_sleep_compressed  --sleep-steps 1500 --sleep-lr 3e-4 --rehearse 1.0 --rehearse-mem 3.0
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
