#!/usr/bin/env bash
# r98 — (a) is fuzzy recall SEMANTIC? synonym-query probe across models; (b) does a softer
# read widen the (approximate-key) basin? settle-beta sweep on the wide model. All eval, no train.
# box1 = 4 arch shapes (semantic) + beta sweep on d512_L2; box2 = plain vs gated (semantic).
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
R=experiment/r98
mkdir -p $R/results $R/logs || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
PROG=$R/logs/_progress_${WHICH}.log
echo "r98 $WHICH starting in $ROOT at $(date -Is)" | tee -a $PROG
SEM () { name=$1; run=$2; ckpt=$3
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  [ -f "$ckpt" ] || { echo "SKIP $name (no ckpt)" >> $PROG; return; }
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.persist semantic --run "$run" --ckpt "$ckpt" \
    --batch 8 --out $R/results/$name.json > $R/logs/$name.log 2>&1
  grep -E "recall|cos" $R/logs/$name.log >> $PROG
}
BETA () { beta=$1; ck=experiment/r87/ckpt/arch2_d512_L2.pt; rn=experiment/r87/results/arch2_d512_L2.json
  nm=beta_${beta}
  [ -f "$R/results/$nm.json" ] && { echo "SKIP $nm" >> $PROG; return; }
  echo "=== $nm START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.persist fuzzy --run "$rn" --ckpt "$ck" \
    --nn-ranks 0 1 2 3 --settle-beta $beta --batch 32 --iters 15 --out $R/results/$nm.json > $R/logs/$nm.log 2>&1
  echo "beta=$beta:" >> $PROG; grep -E "nn_rank=" $R/logs/$nm.log >> $PROG
}
if [ "$WHICH" = "box1" ]; then
  for m in d256_L6 d512_L2 d384_L3 d192_L12; do
    SEM sem_$m experiment/r87/results/arch2_$m.json experiment/r87/ckpt/arch2_$m.pt
  done
  for b in 8 16 32 64 128; do BETA $b; done
else
  SEM sem_recall_exact experiment/r92/results/r92_recall_exact_cold.json experiment/r92/ckpt/r92_recall_exact_cold.pt
  SEM sem_store_f64     experiment/r85/results/store_f64_m2048.json       experiment/r85/ckpt/store_f64_m2048.pt
fi
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -30 $PROG