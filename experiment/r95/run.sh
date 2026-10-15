#!/usr/bin/env bash
# r95 — the HONEST CoT-over-memory number: autoregressive multi-hop eval (persist chain_ar)
# on the trained chain checkpoints, feeding the model its OWN emitted intermediate per hop.
# box1 = 4-hop (r93) + 5-hop (r94); box2 = 2-hop@64 (r93) + 3-hop@64 (r94). Fast (eval only).
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
R=experiment/r95
mkdir -p $R/results $R/logs || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
PROG=$R/logs/_progress_${WHICH}.log
echo "r95 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
AR () { name=$1; run=$2; ckpt=$3
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.persist chain_ar --run "$run" --ckpt "$ckpt" \
    --batch 32 --iters 20 --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  tail -1 $R/logs/$name.log >> $PROG
}
if [ "$WHICH" = "box1" ]; then
  AR ar_4hop experiment/r93/results/r93_chain_4hop.json experiment/r93/ckpt/r93_chain_4hop.pt
  AR ar_5hop experiment/r94/results/r94_chain_5hop.json experiment/r94/ckpt/r94_chain_5hop.pt
else
  AR ar_2hop_f64 experiment/r93/results/r93_chain_2hop_f64.json experiment/r93/ckpt/r93_chain_2hop_f64.pt
  AR ar_3hop_f64 experiment/r94/results/r94_chain_3hop_f64.json experiment/r94/ckpt/r94_chain_3hop_f64.pt
fi
echo "=== ROUND DONE $(date -Is)" >> $PROG