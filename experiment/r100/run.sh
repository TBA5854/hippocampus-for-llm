#!/usr/bin/env bash
# r100 — growable store (append like a KV cache) removes the r51 ring cliff. Context-bridge
# probe on a plain recall model: fixed ring vs --store-grow, sweeping the bridge distance far
# past r62's 63 windows. Eval only, no training.
# usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework"; RUN=experiment/r87/results/arch2_d512_L2.json; CK=experiment/r87/ckpt/arch2_d512_L2.pt ;;
  box2) ROOT="$HOME/fw";                 RUN=experiment/r87/results/arch2_d256_L6.json; CK=experiment/r87/ckpt/arch2_d256_L6.pt ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r100
mkdir -p $R/results $R/logs || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
PROG=$R/logs/_progress_${WHICH}.log
echo "r100 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
[ -f "$CK" ] || { echo "no ckpt $CK; abort" | tee -a $PROG; exit 1; }
NSEG="4 8 16 32 64 128 200"
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.persist context --run "$RUN" --ckpt "$CK" \
    --mem-list 512 --nseg-list $NSEG --batch 8 --iters 8 --out $R/results/$name.json "$@" \
    > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -E "M=|recall" $R/logs/$name.log >> $PROG
}
run ctx_fixed
run ctx_grow --store-grow
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -20 $PROG