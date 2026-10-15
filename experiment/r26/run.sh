#!/usr/bin/env bash
# r26 (box1) — can a stored value be decoded back to a token id? Prerequisite for sleep-as-eviction.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r26
mkdir -p $R/logs $R/results

echo "=== r26_decode_8x START $(date -Is)" >> $R/logs/_progress.log
uv run --group sleep python -u -m mech.decode \
    --run experiment/r20/results/r20_warm_skipnorm_M4096.json \
    --ckpt experiment/r20/ckpt/r20_warm_skipnorm_M4096.pt \
    --batch 4 --out $R/results/r26_decode_8x.json > $R/logs/r26_decode_8x.log 2>&1
echo "=== r26_decode_8x EXIT $? $(date -Is)" >> $R/logs/_progress.log

# the 1x/48-fact checkpoint lives on box2 (~/fw); it is copied in before this runs
if [ -f experiment/r25/ckpt/r25_1x_f48_warm.pt ]; then
  echo "=== r26_decode_1x_f48 START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.decode \
      --run experiment/r25/results/r25_1x_f48_warm.json \
      --ckpt experiment/r25/ckpt/r25_1x_f48_warm.pt \
      --batch 8 --out $R/results/r26_decode_1x_f48.json > $R/logs/r26_decode_1x_f48.log 2>&1
  echo "=== r26_decode_1x_f48 EXIT $? $(date -Is)" >> $R/logs/_progress.log
else
  echo "=== r26_decode_1x_f48 SKIPPED (no checkpoint) $(date -Is)" >> $R/logs/_progress.log
fi
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
