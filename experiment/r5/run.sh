#!/usr/bin/env bash
# r5 — break key-vs-key confusion at 8 facts: sharper surrogate, identity keys, both.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r5
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --n-facts 8 --answer-weight 5.0 --batch 32 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinyshakespeare.txt --dg-bypass --eval-every 250
      --lr 0.0003 --memory-size 512 --n-heads 1 --n-layers 6 --n-seg 4 --readout graded
      --seg-len 128 --steps 4000 --seed 0 --write-threshold 0.6
      --write-layer 5 --value-shift 0 --query-init identity"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

# Reordered after r4_f8_s1 collapsed (alignment 30 -> 0%, gate never opened):
# seed robustness is now the decisive question, so the fixes run on the failing
# seed first. A later --seed overrides BASE's --seed 0 (argparse: last wins).
run r5_k0          --beta-grad 4  --key-layer 0 --read-layers 0
run r5_k0_s1       --beta-grad 4  --key-layer 0 --read-layers 0 --seed 1
run r5_floor_s1    --beta-grad 4  --key-layer 1 --read-layers 1 --read-grad-floor 0.1 --seed 1
run r5_bg16        --beta-grad 16 --key-layer 1 --read-layers 1
run r5_bg16_k0     --beta-grad 16 --key-layer 0 --read-layers 0
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
