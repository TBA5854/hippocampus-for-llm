#!/usr/bin/env bash
# r35 — language: facts stated in text, queried in text, PG-19 prose as distractors.
#
# BASE is r23/run.sh verbatim (a round that reached 99.7%), changing only the
# task, the corpus and the sizes. r34 cost a full set of runs by taking the
# placement from the plan's prose instead of from a working run.sh; this is that
# lesson applied. In particular --value-skip-layer 0 --value-skip-norm is kept,
# which r17/r18 proved the last-block write requires and r34 did not have.
#
# Three metrics per eval: exact (chance 0.15%), char (chance 3.8%), ce (chance
# 3.258). A positive needs exact; a negative needs all three, or it cannot
# separate "the store does not carry this" from "the ruler is too coarse" --
# which is exactly how --task recall was abandoned before r1.
#
#   box1: ROOT=~/research/framework   box2: ROOT=~/fw
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT" || { echo "FATAL: cannot cd to ROOT=$ROOT" >&2; exit 1; }
R=experiment/r35
# Fail loudly on the wrong ROOT. On box2 ~/research/framework belongs to another
# account: it is readable, so `cd` succeeds, but every write fails. With set -u
# and no set -e that cascaded silently and the round produced an empty log and
# no runs at all -- it looked launched and was not.
mkdir -p $R/logs $R/results $R/ckpt \
  || { echo "FATAL: ROOT=$ROOT is not writable (box2 needs ROOT=\$HOME/fw)" >&2; exit 1; }
[ -w "$R/logs" ] || { echo "FATAL: $ROOT/$R/logs not writable" >&2; exit 1; }
echo "r35 starting in $ROOT at $(date -Is)"

BASE="--task recall --hc on --n-facts 8 --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data-books data/pg19 --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 0
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0 --value-skip-norm
      --memory-size 512 --n-seg 4 --batch 32 --steps 6000"

run () {   # run <name> <extra args...>   — sequential, logged, checkpointed
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt \
      > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

case "${1:?usage: run.sh pair|controls}" in
  pair)                                    # box1: the runs that decide it
    run r35_on   --hc on
    run r35_off  --hc off
    ;;
  controls)                                # box2: ceiling first, it gates the rest
    run r35_ceiling --facts-in-context
    run r35_shuf    --shuffle-store
    ;;
esac
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
