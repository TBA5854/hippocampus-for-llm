#!/usr/bin/env bash
# r96 — seed replication of the deep multi-hop headline points (paper robustness):
# box1 = 4-hop @32 seed 1, box2 = 3-hop @64 seed 1, each with a broken-chain control.
# Warm from r79 (r84 recipe). Confirms r93/r94's seed-0 results replicate.
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
R=experiment/r96
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
BASE=experiment/r79/ckpt/r79_shared_gate_recall.pt
COMMON="--task bpe_chain --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity --n-steps 0 --beta 32
      --n-seg 4 --batch 32 --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5
      --value-skip-layer 0 --value-skip-norm --recency-read --write-op-gate --write-skip-pad
      --n-chains 4 --init $BASE --seed 1"
PROG=$R/logs/_progress_${WHICH}.log
echo "r96 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
if [ "$WHICH" = "box1" ]; then
  run r96_chain_4hop_s1        --n-hops 4 --n-facts 32 --memory-size 1024 --steps 8000
  run r96_chain_4hop_s1_broken --n-hops 4 --n-facts 32 --memory-size 1024 --steps 3000 --broken-chain
else
  run r96_chain_3hop_f64_s1        --n-hops 3 --n-facts 64 --memory-size 2048 --steps 8000
  run r96_chain_3hop_f64_s1_broken --n-hops 3 --n-facts 64 --memory-size 2048 --steps 3000 --broken-chain
fi
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG