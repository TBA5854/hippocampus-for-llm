#!/usr/bin/env bash
# r56 — placement matrix: read @ start/all x write @ end/all (MQAR, r11 config).
# Phase 1: main/ctrl (write-side, read@0). Phase 2: read_main/read_ctrl (read@all).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r56
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
SFX="${SFX:-}"                       # box2 sets SFX=_box2 for its log names
BASE="--task mqar --hc on --no-tile --n-facts 8 --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinystories.txt --dg-bypass --eval-every 250
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --value-shift 1 --n-steps 0 --beta 32 --value-skip-layer 0
      --memory-size 512 --n-seg 4 --batch 32 --steps 6000"
CASE="${1:?usage: run.sh main|ctrl|read_main|read_ctrl}"
case "$CASE" in
  read_*) PROG=$R/logs/_progress_read${SFX}.log ;;
  *)      PROG=$R/logs/_progress${SFX}.log ;;
esac
echo "r56 $CASE starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $BASE "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
case "$CASE" in
  # --- phase 1: read @ layer 0, vary the write ---
  main)
    run r56_wlast --seed 0 --read-layers 0 --write-layer 5
    run r56_wall  --seed 0 --read-layers 0 --write-layers 0 1 2 3 4 5
    ;;
  ctrl)
    run r56_wall_shuf --seed 0 --read-layers 0 --write-layers 0 1 2 3 4 5 --shuffle-store
    run r56_wall_s1   --seed 1 --read-layers 0 --write-layers 0 1 2 3 4 5
    ;;
  # --- phase 2: read @ every layer, vary the write ---
  read_main)
    run r56_rall_wlast --seed 0 --read-layers 0 1 2 3 4 5 --write-layer 5
    run r56_rall_wall  --seed 0 --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5
    ;;
  read_ctrl)
    run r56_rall_wall_shuf --seed 0 --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5 --shuffle-store
    run r56_rall_wall_s1   --seed 1 --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5
    ;;
esac
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
