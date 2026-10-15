#!/usr/bin/env bash
# r86 — memory transplant. Two independent d256L6 recall models, then the swap eval.
# usage: run.sh box1|box2   (launch detached: setsid bash experiment/r86/run.sh box1 ...)
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework" ;;
  box2) ROOT="$HOME/fw" ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r86
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
# plain store (value-skip-norm, NO write-op-gate) so both models cold-bootstrap (r66)
COMMON="--task bpe_recall --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --n-steps 0 --beta 32 --memory-size 512 --n-seg 4 --batch 32 --n-facts 16
      --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5 --value-skip-layer 0 --value-skip-norm"
PROG=$R/logs/_progress_${WHICH}.log
echo "r86 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
run r86_modelA --seed 0 --steps 12000
run r86_modelB --seed 1 --steps 12000
echo "=== swap eval $(date -Is)" >> $PROG
uv run --group sleep python -u -m mech.persist swap \
  --run $R/results/r86_modelA.json --ckpt $R/ckpt/r86_modelA.pt \
  --run-b $R/results/r86_modelB.json --ckpt-b $R/ckpt/r86_modelB.pt \
  --batch 32 --out $R/results/swap.json > $R/logs/swap.log 2>&1
echo "=== swap EXIT $? $(date -Is)" >> $PROG
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG; cat $R/results/swap.json 2>/dev/null