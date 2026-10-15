#!/usr/bin/env bash
# r28 (box1, second pass) — the 8x comparison at a bigger eval. batch 32 OOMs (15.6 GB; memory scales with n_seg x M x batch), so 16. The first pass ran --batch 8,
# so n=8 queries: one answer was worth 12.5 points and "old store 100%" could not be trusted.
# The 1x sweep (n=64) showed memory rehearsal HURTING old-store readability (88% -> 38%),
# which n=8 cannot resolve either way.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd ~/research/framework
R=experiment/r28
mkdir -p $R/logs $R/results

RUN=experiment/r20/results/r20_warm_skipnorm_M4096.json
CK=experiment/r20/ckpt/r20_warm_skipnorm_M4096.pt

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.sleep_evict --run $RUN --ckpt $CK \
      --batch 16 --shared-facts --tag $name "$@" \
      --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r28_8x_n16_mem0 --sleep-steps 1500 --sleep-lr 3e-4 --rehearse 1.0
run r28_8x_n16_mem1 --sleep-steps 1500 --sleep-lr 3e-4 --rehearse 1.0 --rehearse-mem 1.0
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
