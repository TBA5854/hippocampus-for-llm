#!/usr/bin/env bash
# r113 (box2/d256) — cross-model mechanism check for the r112 write-order decay.
# Same arms as box1/d512 run.sh but on the d256/6-layer (r92) model on ~/fw. d256's decay
# is near-binary (only first fact survives) -> does --rare-keys RESCUE late needles (gate)?
# outputs prefixed d256_ so they don't clash with box1's d512 r113 files when pulled local.
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="$HOME/fw"; RUN=experiment/r92/results/r92_recall_exact_cold.json; CK=experiment/r92/ckpt/r92_recall_exact_cold.pt
cd "$ROOT" || { echo "FATAL cd" >&2; exit 1; }
R=experiment/r113; mkdir -p $R/results $R/logs || exit 1
PROG=$R/logs/_progress_box2.log
[ -f "$CK" ] || { echo "no ckpt $CK" | tee -a $PROG; exit 1; }
echo "r113 box2/d256 start $(date -Is)" | tee $PROG

sweep () { name=$1; ns=$2; grid=$3; shift 3; extra="$*"
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  MEM=$(( ns*130 + 10000 ))
  echo "=== $name (nseg $ns M=$MEM) [${extra:-baseline}] START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.persist passkey --run "$RUN" --ckpt "$CK" \
    --mem-list $MEM --nseg-list $ns --batch 2 --iters 8 --depth-list $grid $extra \
    --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -E "depth=|OOM|Error" $R/logs/$name.log | tail -8 >> $PROG
}

COARSE="0.0 0.25 0.5 0.75 1.0"
FINE="0.0 0.1 0.25 0.5 0.75 0.9 1.0"
sweep d256_sat_ctx32   33  "$COARSE"
sweep d256_sat_ctx128  129 "$COARSE"
sweep d256_sat_ctx512  513 "$FINE"
sweep d256_rarekeys_ctx512 513 "$FINE" --rare-keys
sweep d256_grow_ctx512     513 "$FINE" --store-grow
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -34 $PROG
