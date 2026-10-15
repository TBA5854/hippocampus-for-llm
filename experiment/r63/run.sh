#!/usr/bin/env bash
# r63 — does the r59 multi-layer escape transfer to REAL LANGUAGE (BPE)?
# On MQAR, read+write@all escaped the fact-count cliff cold (r59). On BPE the only cold escape so far
# was the r25/r49 curriculum. Test: single-layer vs multi-layer, BPE, cold, at the hard fact counts.
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r63
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
SFX="${SFX:-}"
BPE=data/pg19/bpe_8192.json
BASE="--task bpe_recall --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --n-steps 0 --beta 32 --value-skip-layer 0 --value-skip-norm
      --memory-size 512 --n-seg 4 --batch 32 --steps 8000"
SL="--read-layers 0 --write-layer 5"                       # single-layer (objective placement)
ML="--read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5"  # read + write at every layer
PROG=$R/logs/_progress${SFX}.log
echo "r63 starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $BASE "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
case "${1:?usage: run.sh sl|ml}" in
  sl)   # single-layer baseline (objective), BPE cold
    run r63_sl_f16 --seed 0 --n-facts 16 $SL
    run r63_sl_f32 --seed 0 --n-facts 32 $SL
    ;;
  ml)   # multi-layer read+write, BPE cold — the escape test
    run r63_ml_f16 --seed 0 --n-facts 16 $ML
    run r63_ml_f32 --seed 0 --n-facts 32 $ML
    ;;
esac
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
