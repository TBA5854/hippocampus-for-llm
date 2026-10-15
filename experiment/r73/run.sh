#!/usr/bin/env bash
# r73 — learned reindex on the shared store, leak-proof multi-hop. Usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework" ;;
  box2) ROOT="$HOME/fw" ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r73
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
COMMON="--data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --n-steps 0 --beta 32 --memory-size 512 --n-seg 4 --batch 32 --n-facts 16
      --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5
      --value-skip-layer 0 --value-skip-norm --recency-read"
HOP="--task bpe_hop --n-hops 2 --hop-key retrieved --n-chains 4 --hop-reindex"
PROG=$R/logs/_progress_${WHICH}.log
echo "r73 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
if [ "$WHICH" = "box2" ]; then
  BASE=experiment/r70/ckpt/r70_sh_recall.pt        # existing shared 1-hop base (seed 0)
  run r73_sh_reindex_hop2   $HOP --seed 0 --steps 10000 --init $BASE
  run r73_sh_reindex_broken $HOP --seed 0 --steps 4000 --broken-chain --init $BASE
else
  # replication at seed 1: train a fresh shared base, then reindex hop2
  run r73_sh_recall_s1      --task bpe_recall --seed 1 --steps 8000
  run r73_sh_reindex_hop2_s1 $HOP --seed 1 --steps 8000 --init experiment/r73/ckpt/r73_sh_recall_s1.pt
fi
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
