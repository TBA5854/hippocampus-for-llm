#!/usr/bin/env bash
# r68 — multi-hop through the shared store. Does read layer L's retrieval, fed as
# layer L+1's key, chain A->B->C? Warm from r66_mlcold_f16. box2 (ROOT=~/fw).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r68
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
INIT=experiment/r66/ckpt/r66_mlcold_f16.pt
BASE="--task bpe_hop --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --n-steps 0 --beta 32 --value-skip-layer 0 --value-skip-norm
      --memory-size 512 --n-seg 4 --batch 32 --n-facts 16 --seed 0 --n-hops 2 --recency-read"
ML="--read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5"
PROG=$R/logs/_progress.log
echo "r68 starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $BASE $ML "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
run r68_hop_retrieved --steps 6000 --hop-key retrieved --init $INIT
run r68_hop_token     --steps 6000 --hop-key token     --init $INIT   # control: no chaining
run r68_hop_broken    --steps 4000 --hop-key retrieved --broken-chain --init $INIT  # control
run r68_hop_residual  --steps 6000 --hop-key residual  --init $INIT   # ablation
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
