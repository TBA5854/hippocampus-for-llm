#!/usr/bin/env bash
# r79 — ONE shared store + gated ops (no per-layer). Usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework" ;;
  box2) ROOT="$HOME/fw" ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r79
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
BASE=experiment/r70/ckpt/r70_sh_recall.pt          # shared 1-hop base (seed 0)
COMMON="--data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --n-steps 0 --beta 32 --memory-size 512 --n-seg 4 --batch 32 --n-facts 16 --seed 0
      --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5
      --value-skip-layer 0 --value-skip-norm --recency-read --write-op-gate"
PROG=$R/logs/_progress_${WHICH}.log
echo "r79 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
if [ "$WHICH" = "box1" ]; then
  run r79_shared_gate_recall --task bpe_recall --steps 10000 --init $BASE
else
  run r79_shared_gate_update --task bpe_update --steps 10000 --init $BASE
fi
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
