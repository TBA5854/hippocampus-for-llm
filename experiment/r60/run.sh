#!/usr/bin/env bash
# r60 — seed-robustness of the r59 multi-layer escape (128 symbols). r59 was single-seed;
# per-layer was non-monotonic (16:92.7 24:71.6 32:100). Confirm the escape isn't seed luck.
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r60
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
SFX="${SFX:-}"
COMMON="--task mqar --hc on --no-tile --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0
        --d-mem 256 --d-model 256 --data data/tinystories.txt --dg-bypass --eval-every 250
        --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
        --key-layer 0 --key-pos-free --freeze-query --query-init identity --n-symbols 128
        --value-shift 1 --n-steps 0 --beta 32 --memory-size 512 --n-seg 4 --batch 32 --steps 6000"
SH="--read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5 --value-skip-layer 0"
PL="--per-layer-store --write-layers 0 1 2 3 4 5 --read-layers 0 1 2 3 4 5"
PROG=$R/logs/_progress${SFX}.log
echo "r60 starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
case "${1:?usage: run.sh sh|pl}" in
  sh)   # shared-symmetric, two more seeds at the two cliff points (fast arm)
    run r60_sh_f16_s1 --seed 1 --n-facts 16 $SH
    run r60_sh_f16_s2 --seed 2 --n-facts 16 $SH
    run r60_sh_f32_s1 --seed 1 --n-facts 32 $SH
    run r60_sh_f32_s2 --seed 2 --n-facts 32 $SH
    ;;
  pl)   # per-layer, one more seed where r59 was noisy (slow arm, 2 runs)
    run r60_pl_f16_s1 --seed 1 --n-facts 16 $PL
    run r60_pl_f24_s1 --seed 1 --n-facts 24 $PL
    ;;
esac
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
