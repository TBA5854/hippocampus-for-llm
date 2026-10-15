#!/usr/bin/env bash
# r47 — BPE token-level recall: does a single-token word-key work COLD at 8 facts,
# where char-level needed a curriculum and topped out ~80%? Uses a pre-built BPE.
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r47
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
echo "r47 starting in $ROOT at $(date -Is)"
BPE=data/pg19/bpe_8192.json
BASE="--task bpe_recall --data-books data/pg19 --bpe $BPE --hc on --n-facts 8
      --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256
      --dg-bypass --eval-every 250 --lr 0.0003 --n-heads 1 --n-layers 6
      --readout graded --seg-len 128 --seed 1 --write-threshold 0.6 --key-layer 0
      --read-layers 0 --key-pos-free --freeze-query --query-init identity --n-steps 0
      --beta 32 --write-layer 5 --value-skip-layer 0 --value-skip-norm
      --memory-size 512 --n-seg 4 --batch 32 --steps 6000"
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}
run r47_bpe_on                             # the number, COLD
run r47_bpe_off      --hc off              # floor
run r47_bpe_shuf     --shuffle-store       # scrambled control
run r47_bpe_ceiling  --facts-in-context    # ceiling
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
