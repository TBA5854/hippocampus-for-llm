#!/usr/bin/env bash
# r108 — the LITERAL 1M-token push, now tractable (this is limitation-#2 item #3, the
# streaming eval). The eval-time write fast path (in-place O(S), not the out-of-place
# one-hot O(M*d)-per-window rewrite) makes a ~1M-slot store buildable in seconds/window;
# a FIXED pre-sized store (no grow doubling transient) + batch 1 + top-1-only bounds read
# memory. Pushes nseg to 7800 (~1M tokens of intervening prose). box1 (d512, r87 ckpt).
set -u
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT="$HOME/research/framework"; cd "$ROOT" || { echo "FATAL cd" >&2; exit 1; }
RUN=experiment/r87/results/arch2_d512_L2.json; CK=experiment/r87/ckpt/arch2_d512_L2.pt
R=experiment/r108; mkdir -p $R/results $R/logs || exit 1
PROG=$R/logs/_progress.log
[ -f "$CK" ] || { echo "no ckpt $CK" | tee -a $PROG; exit 1; }
echo "r108 1M push (fast-write, fixed-M, batch1, top-1) start $(date -Is)" | tee $PROG
sp () { ns=$1
  name=normal_ns${ns}
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  MEM=$(( ns*130 + 10000 ))          # fixed, pre-sized to final slots (~127/window)
  echo "=== $name (nseg $ns, fixed M=$MEM) START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.persist sparse --run "$RUN" --ckpt "$CK" \
    --mem-list $MEM --nseg-list $ns --topk-list 1 --batch 1 --iters 4 \
    --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -E "top-|OOM|Error|MemoryError" $R/logs/$name.log | tail -2 >> $PROG
}
for ns in 1000 3000 7800; do sp $ns; done
echo "=== 1M PUSH DONE $(date -Is)" >> $PROG
tail -20 $PROG
