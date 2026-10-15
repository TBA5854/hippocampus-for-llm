#!/usr/bin/env bash
# r88 — capacity curriculum: is the r85 64->128-fact wall under-training or a real cap?
# Warm the 128-fact/M4096 store from the CONVERGED 64-fact model (a curriculum rung),
# with a longer budget. If it now converges, the r85 26% was bootstrapping; if it stays
# stuck, 128 facts is a genuine capacity wall for the single shared managed store.
# usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework" ;;
  box2) ROOT="$HOME/fw" ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r88
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
F64=experiment/r85/ckpt/store_f64_m2048.pt          # converged 64-fact managed store
COMMON="--task bpe_recall --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --n-steps 0 --beta 32 --n-seg 4 --batch 32
      --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5
      --value-skip-layer 0 --value-skip-norm --recency-read --write-op-gate --write-skip-pad"
PROG=$R/logs/_progress_${WHICH}.log
echo "r88 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
# curriculum rung: 64 -> 128 facts, warm from the converged 64-fact store, longer budget
run r88_f128_from_f64  --memory-size 4096 --n-facts 128 --steps 16000 --seed 0 --init $F64
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG