#!/usr/bin/env bash
# r58 — does the r25 curriculum rescue the breaking point for the NEW variants?
# box1: per-layer store curriculum (warm 16 from pl_f8, 32 from pl_16).
# box2: shared symmetric curriculum, self-contained (train 8, warm 16 from it).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r58
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
SFX="${SFX:-}"
COMMON="--task mqar --hc on --no-tile --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
        --d-mem 256 --d-model 256 --data data/tinystories.txt --dg-bypass --eval-every 250
        --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
        --key-layer 0 --key-pos-free --freeze-query --query-init identity
        --value-shift 1 --n-steps 0 --beta 32 --memory-size 512 --n-seg 4 --batch 32 --steps 6000"
PL="--per-layer-store --write-layers 0 1 2 3 4 5 --read-layers 0 1 2 3 4 5"
SH="--read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5 --value-skip-layer 0"
PROG=$R/logs/_progress${SFX}.log
echo "r58 starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
case "${1:?usage: run.sh pl|sh}" in
  pl)   # per-layer curriculum, warm-start from r57's 8-fact per-layer model
    run r58_pl_f16w --seed 0 --n-facts 16 $PL --init experiment/r57/ckpt/r57_pl_f8.pt
    run r58_pl_f32w --seed 0 --n-facts 32 $PL --init experiment/r58/ckpt/r58_pl_f16w.pt
    ;;
  sh)   # shared curriculum, self-contained base then warm
    run r58_sh_f8    --seed 0 --n-facts 8  $SH
    run r58_sh_f16w  --seed 0 --n-facts 16 $SH --init experiment/r58/ckpt/r58_sh_f8.pt
    run r58_sh_f32w  --seed 0 --n-facts 32 $SH --init experiment/r58/ckpt/r58_sh_f16w.pt
    ;;
esac
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
