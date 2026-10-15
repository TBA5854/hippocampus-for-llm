#!/usr/bin/env bash
# r104 — does the r103 per-layer-write win (the new default) move the r88 ~128-fact CAPACITY WALL?
# Matched A/B (baseline read-all/write-MEAN vs spl_token per-layer-write) at higher fact counts.
#   box1 = 128 facts (at the old wall): baseline_mean_f128 vs spl_token_f128
#   box2 =  64 facts (mid):             baseline_mean_f64  vs spl_token_f64
# Cold bpe_recall, r103 hyperparameters, n_seg 8 (room for the extra facts). usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework"; NF=128 ;;
  box2) ROOT="$HOME/fw";                 NF=64  ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r104
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
STEPS=10000
COMMON="--data-books data/pg19 --bpe $BPE --hc on --task bpe_recall --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity --n-steps 0 --beta 32
      --memory-size 512 --n-seg 8 --batch 32 --n-facts $NF --store-grow --steps $STEPS --seed 0"
PROG=$R/logs/_progress_${WHICH}.log
echo "r104 $WHICH (NF=$NF) starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -iE "recall" $R/logs/$name.log | tail -2 >> $PROG
}
# baseline: read every layer, write the MEAN = one slot/token (old default)
BASE="--read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5 --value-skip-layer 0 --value-skip-norm"
run r104_baseline_mean_f${NF} $BASE
# spl_token: per-layer write, exact token key (r103 new default)
run r104_spl_token_f${NF}     --shared-per-layer --shared-pl-key token
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -20 $PROG
