#!/usr/bin/env bash
# r54 — margin curve: how much M does surprisal need to reach the oracle. OFFLINE.
# box2=main(64f), box1=cross(16f seed2).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r54
mkdir -p $R/logs $R/results || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
UV="uv run --group sleep python -u -m mech.persist"
POL="ring oracle surprisal"
MEM="256 384 512 768 1024 1536 2048"
NSEG="16 32"

case "${1:?usage: run.sh main|cross}" in
  main)   TAG=main;  RUN=experiment/r50/results/r50_bpe64.json;    CK=experiment/r50/ckpt/r50_bpe64.pt ;;
  cross)  TAG=cross; RUN=experiment/r50/results/r50_bpe16_s2.json; CK=experiment/r50/ckpt/r50_bpe16_s2.pt ;;
esac
LOG=$R/logs/r54_${TAG}.log
PROG=$R/logs/_progress_${TAG}.log
echo "r54 $TAG starting in $ROOT at $(date -Is)" | tee $PROG
{
  echo "=== r54 $TAG margin curve START $(date -Is)"
  $UV evict --run $RUN --ckpt $CK --iters 10 --batch 16 \
      --policies $POL --mem-list $MEM --nseg-list $NSEG \
      --out $R/results/r54_${TAG}.json
  echo "=== ROUND DONE $(date -Is)"
} > $LOG 2>&1
tail -1 $LOG
