#!/usr/bin/env bash
# r36 — box1: where r35 loses the answer (per-position split) + seeds 1,2.
#       box2: does --suppress-empty-read recover r34's lost net benefit?
#
# box1 BASE is r35's verbatim (which is r23's). box2 BASE is r34's, scored
# against r34_off_s{N} -- same seed, same steps, only the new flag differs, so
# no new control is needed.
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT" || { echo "FATAL: cannot cd to ROOT=$ROOT" >&2; exit 1; }
R=experiment/r36
mkdir -p $R/logs $R/results $R/ckpt \
  || { echo "FATAL: ROOT=$ROOT not writable (box2 needs ROOT=\$HOME/fw)" >&2; exit 1; }
echo "r36 starting in $ROOT at $(date -Is)"

LANG_BASE="--task recall --hc on --n-facts 8 --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data-books data/pg19 --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --value-shift 1 --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0 --value-skip-norm
      --memory-size 512 --n-seg 4 --batch 32 --steps 6000"

# r34's LM arm, unchanged except the new flag (so it stays comparable to r34_off_s*)
LM_BASE="--task lm --data-books data/pg19 --n-seg 4 --seg-len 256 --batch 24
      --steps 10000 --eval-every 250 --hc on --read-layers 0 --key-layer 0 --key-pos-free
      --query-init identity --freeze-query --write-layer 5 --memory-size 1024
      --beta-grad 4 --dg-bypass"

run () {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt \
      > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

case "${1:?usage: run.sh lang|emptyread}" in
  lang)                                  # box1: the per-position split, 2 seeds
    run r36_lang_s1 $LANG_BASE --seed 1
    run r36_lang_s2 $LANG_BASE --seed 2
    ;;
  emptyread)                             # box2: the r34 fix, on the seeds that paid
    run r36_noempty_s1 $LM_BASE --seed 1 --suppress-empty-read
    run r36_noempty_s0 $LM_BASE --seed 0 --suppress-empty-read
    ;;
esac
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
