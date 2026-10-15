#!/usr/bin/env bash
# r38 — control the 4-fact result, then apply r25's curriculum to the language task.
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT" || { echo "FATAL: cannot cd to ROOT=$ROOT" >&2; exit 1; }
R=experiment/r38
mkdir -p $R/logs $R/results $R/ckpt \
  || { echo "FATAL: ROOT=$ROOT not writable (box2 needs ROOT=\$HOME/fw)" >&2; exit 1; }
echo "r38 starting in $ROOT at $(date -Is)"

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

# H38a FIRST: everything else depends on the 4-fact result being real
run r38_f4_shuf --n-facts 4  --value-shift 1 --shuffle-store
# H38b: r25's curriculum, on the language task
run r38_curr8   --n-facts 8  --value-shift 1 --init experiment/r37/ckpt/r37_f4.pt
# H38c prep: a second point on the shift-3 axis, and the warm start for later
run r38_f4_s3   --n-facts 4  --value-shift 3
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
