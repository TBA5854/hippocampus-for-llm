#!/usr/bin/env bash
# r46 — finish the char-level language curve: 12 facts to 24k (ceiling), 13 facts (task cap).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r46
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
echo "r46 starting in $ROOT at $(date -Is)"
BASE="--task recall --hc on --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data-books data/pg19 --dg-bypass --eval-every 1000
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 1
      --write-threshold 0.6 --key-layer 0 --read-layers 0 --key-pos-free --freeze-query
      --query-init identity --n-steps 0 --beta 32 --write-layer 5 --value-skip-layer 0
      --value-skip-norm --memory-size 512 --n-seg 4 --batch 32"
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}
case "${1:?usage: run.sh ceil12|cap13}" in
  ceil12) run r46_curr12_24k --steps 24000 --n-facts 12 --init experiment/r45/ckpt/r45_curr12_from12k.pt ;;
  cap13)  run r46_curr13_12k --steps 12000 --n-facts 13 --init experiment/r45/ckpt/r45_curr12_from12k.pt ;;
esac
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
