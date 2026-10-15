#!/usr/bin/env bash
# r93 — multi-hop frontier: how DEEP (4 hops) and how many FACTS (64) does CoT-over-memory
# reach? r84/r85 did 2-hop/3-hop at 16/32 facts. box1 = depth (4-hop @32), box2 = fact
# count (2-hop @64). Matched broken-chain controls. Warm from r79 (r84 recipe).
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
R=experiment/r93
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
BASE=experiment/r79/ckpt/r79_shared_gate_recall.pt
COMMON="--task bpe_chain --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity --n-steps 0 --beta 32
      --n-seg 4 --batch 32 --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5
      --value-skip-layer 0 --value-skip-norm --recency-read --write-op-gate --write-skip-pad
      --n-chains 4 --init $BASE"
PROG=$R/logs/_progress_${WHICH}.log
echo "r93 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
if [ "$WHICH" = "box1" ]; then   # DEPTH: 4-hop @32 facts
  run r93_chain_4hop        --n-hops 4 --n-facts 32 --memory-size 1024 --steps 8000 --seed 0
  run r93_chain_4hop_broken --n-hops 4 --n-facts 32 --memory-size 1024 --steps 3000 --seed 0 --broken-chain
else                             # FACT COUNT: 2-hop @64 facts
  run r93_chain_2hop_f64        --n-hops 2 --n-facts 64 --memory-size 2048 --steps 8000 --seed 0
  run r93_chain_2hop_f64_broken --n-hops 2 --n-facts 64 --memory-size 2048 --steps 3000 --seed 0 --broken-chain
fi
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG