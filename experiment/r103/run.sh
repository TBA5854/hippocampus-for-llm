#!/usr/bin/env bash
# r103 — per-layer read+write into ONE shared store (new default) vs read-all/write-MEAN.
# Same model/task/steps; only the write scheme (mean-of-layers -> each layer its own value)
# and the key source change. Cold bpe_recall, 32 facts, r92 hyperparameters.
#   box1 = baseline_mean  vs  spl_token    (per-layer write, shared token key)
#   box2 = baseline_mean  vs  spl_residual (per-layer write, per-depth residual key)
# NOTE: both boxes must have the r103 mech code (shared_per_layer). Sync mech/ before launch.
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
R=experiment/r103
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
STEPS=10000
COMMON="--data-books data/pg19 --bpe $BPE --hc on --task bpe_recall --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity --n-steps 0 --beta 32
      --memory-size 512 --n-seg 4 --batch 32 --n-facts 32 --store-grow --steps $STEPS --seed 0"
PROG=$R/logs/_progress_${WHICH}.log
echo "r103 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -iE "best|final|recall" $R/logs/$name.log | tail -3 >> $PROG
}
# baseline: read every layer, write the MEAN of every layer = ONE slot/token (current default).
BASE="--read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5 --value-skip-layer 0 --value-skip-norm"
run r103_baseline_mean $BASE
# SPL: per-layer write into the shared store. read_layers auto-expands to all 6. No value-skip
# needed (layer 0's own slot IS the identity-bearing early value).
if [ "$WHICH" = "box1" ]; then
  run r103_spl_token    --shared-per-layer --shared-pl-key token
else
  run r103_spl_residual --shared-per-layer --shared-pl-key residual
fi
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -24 $PROG
