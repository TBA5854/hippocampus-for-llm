#!/usr/bin/env bash
# r70 — per-layer multistore vs shared store on multi-hop. Usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework" ;;
  box2) ROOT="$HOME/fw" ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r70
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
COMMON="--data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --n-steps 0 --beta 32 --memory-size 512 --n-seg 4 --batch 32 --n-facts 16 --seed 0
      --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5"
if [ "$WHICH" = "box1" ]; then
  MODE="--per-layer-store"                 # multistore: no value-skip (per-layer values are raw)
  TAG=pl
else
  MODE="--value-skip-layer 0 --value-skip-norm --recency-read"   # shared, proven config
  TAG=sh
fi
PROG=$R/logs/_progress_${WHICH}.log
echo "r70 $WHICH ($TAG) starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON $MODE "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
# stage A: base 1-hop recall; stage B: 2-hop chain, warm from the base
run r70_${TAG}_recall --task bpe_recall --steps 8000
run r70_${TAG}_hop2   --task bpe_hop --n-hops 2 --hop-key retrieved --steps 6000 \
    --init experiment/r70/ckpt/r70_${TAG}_recall.pt
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
