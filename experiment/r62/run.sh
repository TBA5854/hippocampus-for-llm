#!/usr/bin/env bash
# r62 — harden the r61 north-star result: matched scrambled control, a 2nd model, a bigger scale.
# OFFLINE sparse-read probe. box2=main(r50_bpe64), box1=cross(r50_bpe16_s2).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r62
mkdir -p $R/logs $R/results || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
SFX="${SFX:-}"
UV="uv run --group sleep python -u -m mech.persist"
PROG=$R/logs/_progress${SFX}.log
echo "r62 starting in $ROOT at $(date -Is)" | tee $PROG
case "${1:?usage: run.sh main|cross}" in
  main)   RUN=experiment/r50/results/r50_bpe64.json;    CK=experiment/r50/ckpt/r50_bpe64.pt;    TAG=bpe64;    SCALES="2048 16 4096 32 8192 64" ;;
  cross)  RUN=experiment/r50/results/r50_bpe16_s2.json; CK=experiment/r50/ckpt/r50_bpe16_s2.pt; TAG=bpe16s2;  SCALES="2048 16 4096 32" ;;
esac
{
  echo "=== r62 $TAG START $(date -Is)"
  set -- $SCALES
  while [ $# -ge 2 ]; do M=$1; N=$2; shift 2
    echo "--- $TAG  M=$M  bridge=$(( N - 1 )) windows  (real) ---"
    $UV sparse --run $RUN --ckpt $CK --iters 10 --batch 16 \
        --mem-list $M --nseg-list $N --topk-list 1 2 4 8 0 \
        --out $R/results/r62_${TAG}_M${M}_n${N}.json
  done
  echo "--- $TAG  M=4096  bridge=31 windows  (SHUFFLE control) ---"
  $UV sparse --run $RUN --ckpt $CK --iters 10 --batch 16 --shuffle \
      --mem-list 4096 --nseg-list 32 --topk-list 1 0 \
      --out $R/results/r62_${TAG}_M4096_n32_shuf.json
  echo "=== ROUND DONE $(date -Is)"
} > $R/logs/r62_${TAG}.log 2>&1
tail -1 $R/logs/r62_${TAG}.log
