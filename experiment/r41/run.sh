#!/usr/bin/env bash
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r41
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
echo "r41 starting in $ROOT at $(date -Is)"
BASE="--task recall --hc on --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data-books data/pg19 --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 1
      --write-threshold 0.6 --key-layer 0 --key-pos-free --freeze-query
      --query-init identity --n-steps 0 --beta 32 --write-layer 5 --value-skip-layer 0
      --value-skip-norm --memory-size 512 --n-seg 4 --batch 32 --steps 6000 --n-facts 4"
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}
# read at layer 0 only (baseline) vs read at every layer — write stays at last block
run r41_read0    --read-layers 0
run r41_readall  --read-layers 0 1 2 3 4 5
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
