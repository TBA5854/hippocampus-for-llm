#!/usr/bin/env bash
# r113 — characterize the r112 WRITE-ORDER decay (early needle 100% -> late 12.5% @64k).
# Discriminators: (a) SATURATION -> sweep ctx 32/128/512 (does decay steepen?);
# (b) NOVELTY WRITE-GATE -> --rare-keys (unique always-novel keys flatten it iff gate);
# (c) ring LAPPING -> --store-grow (never laps; should NOT change -> rules lapping out).
# top-1, batch 2, iters 8 (n=16 de-noised). box1/d512 (r87). usage: run.sh
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="$HOME/research/framework"; RUN=experiment/r87/results/arch2_d512_L2.json; CK=experiment/r87/ckpt/arch2_d512_L2.pt
cd "$ROOT" || { echo "FATAL cd" >&2; exit 1; }
R=experiment/r113; mkdir -p $R/results $R/logs || exit 1
PROG=$R/logs/_progress.log
[ -f "$CK" ] || { echo "no ckpt $CK" | tee -a $PROG; exit 1; }
echo "r113 start $(date -Is)" | tee $PROG

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
# (a) saturation scaling
sweep sat_ctx32   33  "$COARSE"
sweep sat_ctx128  129 "$COARSE"
sweep sat_ctx512  513 "$FINE"                 # the reference decay, fine grid
# (b) write-gate discriminator + (c) lapping control, at the high-saturation point
sweep rarekeys_ctx512 513 "$FINE" --rare-keys
sweep grow_ctx512     513 "$FINE" --store-grow
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -34 $PROG
