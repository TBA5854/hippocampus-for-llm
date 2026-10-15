#!/usr/bin/env bash
# r20 persistence at the 8x store: fill a store, save it, then answer from it in a FRESH process.
# --batch 2 because each example carries 32 segments x 4096 slots (r19 used --batch 8 at 4x).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd ~/research/framework
R=experiment/r20
mkdir -p $R/offline
RUN=$R/results/r20_warm_skipnorm_M4096.json
CK=$R/ckpt/r20_warm_skipnorm_M4096.pt
D=$R/offline/store

{
  uv run --group sleep python -u -m mech.persist write --run $RUN --ckpt $CK --dir $D \
      --batch 2 --iters 20 --out $R/offline/write.json
  uv run --group sleep python -u -m mech.persist read  --run $RUN --ckpt $CK --dir $D \
      --batch 2 --iters 20 --out $R/offline/read.json
  uv run --group sleep python -u -m mech.persist read  --run $RUN --ckpt $CK --dir $D \
      --batch 2 --iters 20 --mismatch --out $R/offline/read_mismatch.json
  echo "=== PERSIST DONE $(date -Is)"
} > $R/logs/persist.log 2>&1
