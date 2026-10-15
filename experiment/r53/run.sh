#!/usr/bin/env bash
# r53 — surprisal-scored eviction vs oracle/ring. OFFLINE. box2=main(64f), box1=cross(16f seed2).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r53
mkdir -p $R/logs $R/results || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
UV="uv run --group sleep python -u -m mech.persist"
POL="ring oracle surprisal"
MEM="256 512"
NSEG="4 8 16 32"

case "${1:?usage: run.sh main|cross}" in
  main)   TAG=main;  RUN=experiment/r50/results/r50_bpe64.json;    CK=experiment/r50/ckpt/r50_bpe64.pt ;;
  cross)  TAG=cross; RUN=experiment/r50/results/r50_bpe16_s2.json; CK=experiment/r50/ckpt/r50_bpe16_s2.pt ;;
esac
LOG=$R/logs/r53_${TAG}.log
PROG=$R/logs/_progress_${TAG}.log
echo "r53 $TAG starting in $ROOT at $(date -Is)" | tee $PROG
{
  echo "=== r53 $TAG evict (ring/oracle/surprisal) START $(date -Is)"
  $UV evict --run $RUN --ckpt $CK --iters 10 --batch 16 \
      --policies $POL --mem-list $MEM --nseg-list $NSEG \
      --out $R/results/r53_${TAG}.json
  echo "=== ROUND DONE $(date -Is)"
} > $LOG 2>&1
tail -1 $LOG
