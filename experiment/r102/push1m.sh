#!/usr/bin/env bash
# r102 1M-token push, box1, MEMORY-FRUGAL. The grow path amortized-doubles GB-sized
# key/value buffers ~10x; the caching allocator fragments and balloons to ~14G, then
# OOMs the next 2G doubling and crawls on CPU. Fix: a FIXED store pre-sized to the final
# slot count in ONE allocation (no doubling, no fragmentation). top-1 only (the headline).
# ns3000 (~0.5M tokens) then ns7800 (~1M tokens). box1 only.
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="$HOME/research/framework"; cd "$ROOT" || { echo "FATAL cd" >&2; exit 1; }
RUN=experiment/r87/results/arch2_d512_L2.json; CK=experiment/r87/ckpt/arch2_d512_L2.pt
R=experiment/r102; mkdir -p $R/results $R/logs || exit 1
PROG=$R/logs/_progress_box1.log
[ -f "$CK" ] || { echo "no ckpt $CK" | tee -a $PROG; exit 1; }
echo "r102 1M push (fixed-M, no grow) box1 starting $(date -Is)" | tee -a $PROG
sp () { name=$1; ns=$2; shift 2
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  MEM=$(( ns*130 + 10000 ))          # fixed, pre-sized to final slots (~127/window)
  echo "=== $name (nseg $ns, FIXED M=$MEM, topk 1) START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.persist sparse --run "$RUN" --ckpt "$CK" \
    --mem-list $MEM --nseg-list $ns --topk-list 1 --batch 2 --iters 4 "$@" \
    --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -E "top-|OOM|Error|MemoryError" $R/logs/$name.log | tail -3 >> $PROG
}
for ns in 3000 7800; do
  sp rare_ns${ns}   $ns --rare-keys
  sp normal_ns${ns} $ns
done
echo "=== 1M PUSH DONE $(date -Is)" >> $PROG
tail -24 $PROG
