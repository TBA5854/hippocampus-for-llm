#!/usr/bin/env bash
# r42 — the user's idea: reuse DG (pattern separation) for the can't-differentiate problem.
# Tests BOTH my caveat (DG no-ops on identical layer-0 keys) and the real version
# (contextual key so colons differ, THEN DG separates them). 8 facts, seed 1.
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r42
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
echo "r42 starting in $ROOT at $(date -Is)"
COMMON="--task recall --hc on --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data-books data/pg19 --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --seed 1
      --write-threshold 0.6 --n-steps 0 --beta 32 --write-layer 5 --value-skip-layer 0
      --value-skip-norm --memory-size 512 --n-seg 4 --batch 32 --steps 6000 --n-facts 8"
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}
# a) DG ON in the CURRENT config (layer-0 pos-free key). My claim: no-ops (~10%), identical keys.
run r42_dg_l0    $COMMON --key-layer 0 --read-layers 0 --key-pos-free --freeze-query --query-init identity
# swap --dg-bypass off for DG on: rebuild COMMON without it
COMMON_DG="${COMMON/--dg-bypass/}"
run r42_dg_l0on  $COMMON_DG --key-layer 0 --read-layers 0 --key-pos-free --freeze-query --query-init identity
# b) contextual key (layer 1, position-carrying) + DG ON: the meaningful version. May drift (r4).
run r42_ctx_dg   $COMMON_DG --key-layer 1 --read-layers 1
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
