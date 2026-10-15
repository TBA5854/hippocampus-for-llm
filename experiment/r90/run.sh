#!/usr/bin/env bash
# r90 — curriculum for the write-op GATE. r85: the gate fails to cold-bootstrap in a deep
# stack (d256/6L, d192/12L at ~0%); r87: the PLAIN store bootstraps those shapes fine
# (84.7%, 97.0%). So warm the GATE from the r87 plain-store checkpoints: does the gate
# learn (recall stays high AND mem drops to O(facts)) when the store is already
# bootstrapped? If yes, a curriculum removes the gate x depth cold-start dependence.
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
R=experiment/r90
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
COMMON="--task bpe_recall --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --dg-bypass --eval-every 500 --lr 0.0003 --n-heads 1
      --readout graded --seg-len 128 --write-threshold 0.6 --key-layer 0 --key-pos-free
      --freeze-query --query-init identity --n-steps 0 --beta 32 --memory-size 1024 --n-seg 4
      --batch 32 --n-facts 32 --value-skip-layer 0 --value-skip-norm --recency-read
      --write-op-gate --write-skip-pad --steps 6000 --seed 0"
PROG=$R/logs/_progress_${WHICH}.log
echo "r90 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
# warm the gate from the r87 plain-store checkpoints of the shapes that failed cold gated
run r90_gate_d256_L6  --d-model 256 --d-mem 256 --n-layers 6  --read-layers 0 1 2 3 4 5             --write-layers 0 1 2 3 4 5             --init experiment/r87/ckpt/arch2_d256_L6.pt
run r90_gate_d192_L12 --d-model 192 --d-mem 192 --n-layers 12 --read-layers 0 1 2 3 4 5 6 7 8 9 10 11 --write-layers 0 1 2 3 4 5 6 7 8 9 10 11 --init experiment/r87/ckpt/arch2_d192_L12.pt
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG