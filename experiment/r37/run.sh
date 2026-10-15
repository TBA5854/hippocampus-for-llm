#!/usr/bin/env bash
# r37 — does pos0 track 1/n_facts, and does --value-shift 3 address around the
# delimiter? BASE is r36's LANG_BASE verbatim (= r35's = r23's).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT" || { echo "FATAL: cannot cd to ROOT=$ROOT" >&2; exit 1; }
R=experiment/r37
mkdir -p $R/logs $R/results $R/ckpt \
  || { echo "FATAL: ROOT=$ROOT not writable (box2 needs ROOT=\$HOME/fw)" >&2; exit 1; }
echo "r37 starting in $ROOT at $(date -Is)"

BASE="--task recall --hc on --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data-books data/pg19 --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 1
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --n-steps 0 --beta 32
      --write-layer 5 --value-skip-layer 0 --value-skip-norm
      --memory-size 512 --n-seg 4 --batch 32 --steps 6000"

run () {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt \
      > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

# H37a: the sweep. r36's 8-fact runs are the middle point.
run r37_f4     --n-facts 4  --value-shift 1
run r37_f12    --n-facts 12 --value-shift 1
# H37b: address around the delimiter, matched to r36_lang_s1 (seed 1, 8 facts)
run r37_shift3 --n-facts 8  --value-shift 3
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
