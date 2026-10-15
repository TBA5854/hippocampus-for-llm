#!/usr/bin/env bash
# r72 — leak-proof multi-hop (n_chains>1) on both stores. Usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework" ;;
  box2) ROOT="$HOME/fw" ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r72
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
COMMON="--task bpe_hop --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --n-steps 0 --beta 32 --memory-size 512 --n-seg 4 --batch 32 --n-facts 16 --seed 0
      --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5
      --n-hops 2 --hop-key retrieved --n-chains 4"
if [ "$WHICH" = "box1" ]; then
  MODE="--per-layer-store"; TAG=pl; BASE=experiment/r70/ckpt/r70_pl_recall.pt
else
  MODE="--value-skip-layer 0 --value-skip-norm --recency-read"; TAG=sh; BASE=experiment/r70/ckpt/r70_sh_recall.pt
fi
PROG=$R/logs/_progress_${WHICH}.log
echo "r72 $WHICH ($TAG) starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON $MODE "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
run r72_${TAG}_hop2_lp   --steps 8000 --init $BASE
run r72_${TAG}_broken_lp --steps 4000 --broken-chain --init $BASE
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
