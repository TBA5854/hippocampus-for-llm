#!/usr/bin/env bash
# r55 — dedup-at-write vs plain surprisal vs oracle. OFFLINE.
# box1=main(16f, the r54 failure), box2=control(64f).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r55
mkdir -p $R/logs $R/results || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
UV="uv run --group sleep python -u -m mech.persist"
POL="oracle surprisal surprisal_dedup"
MEM="256 384 512 768 1024 1536 2048"
NSEG="16 32"

case "${1:?usage: run.sh main|control}" in
  main)     TAG=main;    RUN=experiment/r50/results/r50_bpe16_s2.json; CK=experiment/r50/ckpt/r50_bpe16_s2.pt ;;
  control)  TAG=control; RUN=experiment/r50/results/r50_bpe64.json;    CK=experiment/r50/ckpt/r50_bpe64.pt ;;
esac
LOG=$R/logs/r55_${TAG}.log
PROG=$R/logs/_progress_${TAG}.log
echo "r55 $TAG starting in $ROOT at $(date -Is)" | tee $PROG
{
  echo "=== r55 $TAG dedup START $(date -Is)"
  $UV evict --run $RUN --ckpt $CK --iters 10 --batch 16 \
      --policies $POL --mem-list $MEM --nseg-list $NSEG \
      --out $R/results/r55_${TAG}.json
  echo "=== ROUND DONE $(date -Is)"
} > $LOG 2>&1
tail -1 $LOG
