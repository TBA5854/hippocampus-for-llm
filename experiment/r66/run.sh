#!/usr/bin/env bash
# r66 — chase the curriculum: does multi-layer read/write reach 100% on BPE with (or without) curriculum?
# box1 mlcurr: ml curriculum ladder (8->16->32 warm). box2 mlcold: ml cold, LONG training (2x steps).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r66
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
SFX="${SFX:-}"
BPE=data/pg19/bpe_8192.json
BASE="--task bpe_recall --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --n-steps 0 --beta 32 --value-skip-layer 0 --value-skip-norm
      --memory-size 512 --n-seg 4 --batch 32"
ML="--read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5"
PROG=$R/logs/_progress${SFX}.log
echo "r66 starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $BASE $ML "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
case "${1:?usage: run.sh mlcurr|mlcold}" in
  mlcurr) # multi-layer + curriculum ladder: does it reach 100% (like single-layer curriculum r49)?
    run r66_mlcurr_f8  --seed 0 --n-facts 8  --steps 8000
    run r66_mlcurr_f16 --seed 0 --n-facts 16 --steps 8000 --init experiment/r66/ckpt/r66_mlcurr_f8.pt
    run r66_mlcurr_f32 --seed 0 --n-facts 32 --steps 8000 --init experiment/r66/ckpt/r66_mlcurr_f16.pt
    ;;
  mlcold) # multi-layer COLD, LONG training: does more training alone remove the curriculum?
    run r66_mlcold_f16 --seed 0 --n-facts 16 --steps 16000
    run r66_mlcold_f32 --seed 0 --n-facts 32 --steps 16000
    ;;
esac
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
