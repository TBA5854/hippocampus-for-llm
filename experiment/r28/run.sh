#!/usr/bin/env bash
# r28 (box1) — does the eviction loop hold at the 8x store (3,937 slots)?
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r28
mkdir -p $R/logs $R/results

RUN=experiment/r20/results/r20_warm_skipnorm_M4096.json
CK=experiment/r20/ckpt/r20_warm_skipnorm_M4096.pt

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.sleep_evict --run $RUN --ckpt $CK \
      --batch 8 --shared-facts --tag $name "$@" \
      --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r28_8x_mem0 --sleep-steps 1500 --sleep-lr 3e-4 --rehearse 1.0
run r28_8x_mem1 --sleep-steps 1500 --sleep-lr 3e-4 --rehearse 1.0 --rehearse-mem 1.0
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
