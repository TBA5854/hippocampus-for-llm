#!/usr/bin/env bash
# r112 — PASSKEY (needle-in-a-haystack): sweep needle DEPTH at fixed context length.
# Hypothesis: content-addressed (position-free layer-0 key) store => recall INVARIANT
# to depth (no lost-in-the-middle), on BOTH models. Control: --shuffle => chance.
# top-1 read, batch 2 (valid shuffle control). usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework"; RUN=experiment/r87/results/arch2_d512_L2.json; CK=experiment/r87/ckpt/arch2_d512_L2.pt ;;
  box2) ROOT="$HOME/fw";                 RUN=experiment/r92/results/r92_recall_exact_cold.json; CK=experiment/r92/ckpt/r92_recall_exact_cold.pt ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r112; mkdir -p $R/results $R/logs || exit 1
PROG=$R/logs/_progress_${WHICH}.log
[ -f "$CK" ] || { echo "no ckpt $CK" | tee -a $PROG; exit 1; }
echo "r112 $WHICH start $(date -Is)" | tee $PROG

# --- smoke: 1 iter, 3 depths, short ctx. Fail fast if the new probe/arg is broken. ---
echo "=== SMOKE $(date -Is)" >> $PROG
if uv run --group sleep python -u -m mech.persist passkey --run "$RUN" --ckpt "$CK" \
     --mem-list 18450 --nseg-list 65 --batch 2 --iters 1 --depth-list 0.0 0.5 1.0 \
     --out /tmp/r112_smoke_${WHICH}.json > $R/logs/_smoke_${WHICH}.log 2>&1; then
  echo "SMOKE OK" >> $PROG; grep "depth=" $R/logs/_smoke_${WHICH}.log >> $PROG
else
  echo "SMOKE FAIL — abort" >> $PROG; tail -12 $R/logs/_smoke_${WHICH}.log >> $PROG; exit 1
fi

pk () { ns=$1; shift; extra="$*"; name=${WHICH}_ctx$((ns-1))${extra:+_ctrl}
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  MEM=$(( ns*130 + 10000 ))
  echo "=== $name (nseg $ns, M=$MEM) ${extra:-sweep} START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.persist passkey --run "$RUN" --ckpt "$CK" \
    --mem-list $MEM --nseg-list $ns --batch 2 --iters 4 $extra \
    --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -E "depth=|OOM|Error" $R/logs/$name.log | tail -6 >> $PROG
}

pk 65               # ctx 64  (~8k tok)  depth sweep
pk 513              # ctx 512 (~64k tok) depth sweep
pk 65 --shuffle     # scrambled control @ ctx 64 -> chance (genuineness)
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -24 $PROG
