#!/usr/bin/env bash
# r111 — extend the 1M result (r108) now that the eval write is O(S)-fast:
#   box1 (d512, r87 ckpt): push to ~2M tokens (nseg 15000). Does top-1 stay flat 100% past 1M?
#   box2 (d256, r92 ckpt): the LITERAL 1M push on the SECOND model (nseg 1000/3000/7800) —
#                          confirms 1M generalizes across models (r102 showed 128k on both).
# fixed pre-sized store, batch 1, top-1 only, normal-word keys. usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework"; RUN=experiment/r87/results/arch2_d512_L2.json; CK=experiment/r87/ckpt/arch2_d512_L2.pt; NSEGS="15000"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ;;
  box2) ROOT="$HOME/fw";                 RUN=experiment/r92/results/r92_recall_exact_cold.json; CK=experiment/r92/ckpt/r92_recall_exact_cold.pt; NSEGS="1000 3000 7800" ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r111; mkdir -p $R/results $R/logs || exit 1
PROG=$R/logs/_progress_${WHICH}.log
[ -f "$CK" ] || { echo "no ckpt $CK" | tee -a $PROG; exit 1; }
echo "r111 $WHICH start $(date -Is)" | tee $PROG
sp () { ns=$1; name=${WHICH}_ns${ns}
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  MEM=$(( ns*130 + 10000 ))
  echo "=== $name (nseg $ns, fixed M=$MEM) START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.persist sparse --run "$RUN" --ckpt "$CK" \
    --mem-list $MEM --nseg-list $ns --topk-list 1 --batch 1 --iters 4 \
    --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -E "top-|OOM|Error" $R/logs/$name.log | tail -2 >> $PROG
}
for ns in $NSEGS; do sp $ns; done
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -16 $PROG
