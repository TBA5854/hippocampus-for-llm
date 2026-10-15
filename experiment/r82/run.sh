#!/usr/bin/env bash
# r82 — route B: loop-gate latent chaining. box2. bpe_hop (target=tail; loop does hops).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r82
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
BASE=experiment/r79/ckpt/r79_shared_gate_recall.pt      # one-store gated, 96% single-hop
COMMON="--task bpe_hop --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --n-steps 0 --beta 32 --memory-size 512 --n-seg 4 --batch 32 --n-facts 16 --seed 0
      --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5
      --value-skip-layer 0 --value-skip-norm --recency-read --write-op-gate
      --n-hops 2 --n-chains 4 --loop-read --loop-iters 2"
PROG=$R/logs/_progress.log
echo "r82 (B) starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
run r82_loop        --steps 8000 --init $BASE
run r82_loop_broken --steps 3000 --broken-chain --init $BASE     # control: must fail
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
