#!/usr/bin/env bash
# r101 — growable store + SPARSE top-1 read together = unbounded store at FLAT recall.
# r100 showed grow removes the ring cliff (but dense read decays with size); r62 showed top-1
# is immune to distractor mass. Here both at once, swept to ~200 windows. Eval only.
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
R=experiment/r101
mkdir -p $R/results $R/logs || { echo "FATAL" >&2; exit 1; }
PROG=$R/logs/_progress_${WHICH}.log
echo "r101 $WHICH starting at $(date -Is)" | tee $PROG
for NS in 16 64 128 200; do
  echo "=== nseg=$NS START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.persist sparse --run "$RUN" --ckpt "$CK" \
    --mem-list 512 --nseg-list $NS --topk-list 1 2 0 --batch 8 --iters 6 --store-grow \
    --out $R/results/grow_sparse_ns${NS}.json > $R/logs/ns${NS}.log 2>&1
  grep -E "top-|graded" $R/logs/ns${NS}.log >> $PROG
done
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -20 $PROG