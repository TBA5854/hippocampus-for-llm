#!/usr/bin/env bash
# r43 — push the winning lever (curriculum) with more training toward the 90% ceiling.
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r43
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
echo "r43 starting in $ROOT at $(date -Is)"
BASE="--task recall --hc on --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data-books data/pg19 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 1
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --n-steps 0 --beta 32 --write-layer 5 --value-skip-layer 0
      --value-skip-norm --memory-size 512 --n-seg 4 --batch 32 --steps 12000"
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}
# does the curriculum reach the 90% (4-fact) ceiling at 8 facts with 2x steps?
run r43_curr8_12k  --n-facts 8 --init experiment/r38/ckpt/r38_curr8.pt
# both levers, longer: shift-3 + curriculum
run r43_s3curr8_12k --n-facts 8 --value-shift 3 --init experiment/r40/ckpt/r40_s3curr8.pt
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
