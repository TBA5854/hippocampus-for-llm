#!/usr/bin/env bash
# r97 — map fuzzy (near-embedding-neighbour) recall across model SHAPE and training.
# persist.py fuzzy, nn-ranks 0..10, iters 20 (n=640). No training. Skips done results.
# box1 = the 4 r87 width x depth shapes; box2 = 2 training variants at d256/6L.
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
R=experiment/r97
mkdir -p $R/results $R/logs || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
PROG=$R/logs/_progress_${WHICH}.log
echo "r97 $WHICH resuming in $ROOT at $(date -Is)" | tee -a $PROG
FZ () { name=$1; run=$2; ckpt=$3
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name (done)" >> $PROG; return; }
  [ -f "$ckpt" ] || { echo "SKIP $name (no ckpt $ckpt)" >> $PROG; return; }
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.persist fuzzy --run "$run" --ckpt "$ckpt" \
    --nn-ranks 0 1 2 3 5 10 --batch 32 --iters 20 --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -E "nn_rank=" $R/logs/$name.log >> $PROG
}
if [ "$WHICH" = "box1" ]; then
  FZ fuzzy_d256_L6  experiment/r87/results/arch2_d256_L6.json  experiment/r87/ckpt/arch2_d256_L6.pt
  FZ fuzzy_d512_L2  experiment/r87/results/arch2_d512_L2.json  experiment/r87/ckpt/arch2_d512_L2.pt
  FZ fuzzy_d384_L3  experiment/r87/results/arch2_d384_L3.json  experiment/r87/ckpt/arch2_d384_L3.pt
  FZ fuzzy_d192_L12 experiment/r87/results/arch2_d192_L12.json experiment/r87/ckpt/arch2_d192_L12.pt
else
  FZ fuzzy_recall_exact experiment/r92/results/r92_recall_exact_cold.json experiment/r92/ckpt/r92_recall_exact_cold.pt
  FZ fuzzy_store_f64    experiment/r85/results/store_f64_m2048.json       experiment/r85/ckpt/store_f64_m2048.pt
fi
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -24 $PROG