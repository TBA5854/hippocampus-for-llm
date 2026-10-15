#!/usr/bin/env bash
# r102 — push the growable store toward 1M-token effective context. Grow + top-1 sparse read,
# swept far in bridge distance (200 -> ~7800 windows = ~1M tokens). Two arms:
#   normal keys  = real words that recur in prose -> maps the collision ceiling (r52/r89).
#   --rare-keys  = rarest words, seldom in prose -> tests whether the STORE itself reaches 1M.
# Eval only. usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework"; RUN=experiment/r87/results/arch2_d512_L2.json; CK=experiment/r87/ckpt/arch2_d512_L2.pt ;;
  box2) ROOT="$HOME/fw";                 RUN=experiment/r92/results/r92_recall_exact_cold.json; CK=experiment/r92/ckpt/r92_recall_exact_cold.pt ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r102
mkdir -p $R/results $R/logs || { echo FATAL >&2; exit 1; }
PROG=$R/logs/_progress_${WHICH}.log
echo "r102 $WHICH starting at $(date -Is)" | tee $PROG
[ -f "$CK" ] || { echo "no ckpt $CK" | tee -a $PROG; exit 1; }
sp () { name=$1; ns=$2; shift 2
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  # top-1 is the headline (flat recall in O(1) slots). The dense graded(all) read
  # materialises ~[M x d] intermediates and OOMs at big M on the shared GPU (ns3000+),
  # then crawls on CPU. So run BOTH only at small M; top-1 ONLY at ns>=3000.
  TK="1 0"; [ "$ns" -ge 3000 ] && TK="1"
  echo "=== $name (nseg $ns, topk $TK) START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.persist sparse --run "$RUN" --ckpt "$CK" \
    --mem-list 512 --nseg-list $ns --topk-list $TK --batch 2 --iters 4 --store-grow "$@" \
    --out $R/results/$name.json > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -E "top-|graded" $R/logs/$name.log >> $PROG
}
for ns in 200 1000 3000 7800; do
  sp rare_ns${ns}   $ns --rare-keys
  sp normal_ns${ns} $ns
done
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -30 $PROG