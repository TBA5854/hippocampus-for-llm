#!/usr/bin/env bash
# r9 — r8 with the query frozen at identity (--freeze-query): same four value sources.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r9
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --no-tile --n-facts 8 --answer-weight 5.0 --batch 32 --beta-grad 4.0
      --beta-max 64.0 --d-mem 256 --d-model 256 --data data/tinyshakespeare.txt --dg-bypass
      --eval-every 250 --lr 0.0003 --memory-size 512 --n-heads 1 --n-layers 6 --n-seg 4
      --readout graded --seg-len 128 --steps 4000 --seed 0 --write-threshold 0.6
      --key-layer 0 --read-layers 0 --key-pos-free --freeze-query --query-init identity"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r9_w0s1 --write-layer 0 --value-shift 1
run r9_w5s1 --write-layer 5 --value-shift 1
run r9_w5s0 --write-layer 5 --value-shift 0
run r9_w3s1 --write-layer 3 --value-shift 1
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
