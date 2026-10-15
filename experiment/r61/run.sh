#!/usr/bin/env bash
# r61 — THE north-star result: store >> context window AND query touches only what's required.
# Fill a big store bridging many windows, read with top-k, sweep k. OFFLINE, r50_bpe64 (box2 ~/fw).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r61
mkdir -p $R/logs $R/results || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
UV="uv run --group sleep python -u -m mech.persist"
RUN=experiment/r50/results/r50_bpe64.json
CK=experiment/r50/ckpt/r50_bpe64.pt
PROG=$R/logs/_progress_box2.log
echo "r61 starting in $ROOT at $(date -Is)" | tee $PROG
{
  echo "=== r61 sparse-read (store >> window, query only top-k) START $(date -Is)"
  # three store scales: 7 / 15 / 31 context windows bridged, top-k in {1,2,4,8, dense}
  for cfg in "1024 8" "2048 16" "4096 32"; do
    set -- $cfg
    echo "--- M=$1  bridge=$(( $2 - 1 )) windows ---"
    $UV sparse --run $RUN --ckpt $CK --iters 10 --batch 16 \
        --mem-list $1 --nseg-list $2 --topk-list 1 2 4 8 0 \
        --out $R/results/r61_M$1_n$2.json
  done
  echo "=== ROUND DONE $(date -Is)"
} > $R/logs/r61_box2.log 2>&1
tail -1 $R/logs/r61_box2.log
