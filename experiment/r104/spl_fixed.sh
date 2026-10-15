#!/usr/bin/env bash
# r103 RE-VERIFY + r104 capacity, after the detach fix. Bug: shared_pl stored pending_values WITH
# the autograd graph, so a later segment's read linked back through every earlier segment's write
# -> graph retained across all n_seg segments -> OOM at n_seg 8 (r104 spl arms crashed), and it may
# have inflated r103 via accidental cross-segment BPTT. Now pending_values are DETACHED (model.py).
#   (1) re-run r103 spl_token (n_seg 4, 32 facts) -> does the +16pt win SURVIVE the fix?
#   (2) run the r104 spl_token capacity arm (n_seg 8).
# box1: seed0 / 128 facts ; box2: seed1 / 64 facts. usage: spl_fixed.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: spl_fixed.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework"; NF=128; SEED=0 ;;
  box2) ROOT="$HOME/fw";                 NF=64;  SEED=1 ;;
  *) echo "usage: spl_fixed.sh box1|box2" >&2; exit 1 ;;
esac
# expandable_segments only on box1 — box2's driver has a broken NVML (nvmlInit assert) that this
# allocator path trips. box2 relies on the detach fix alone (the real memory fix) to fit.
[ "$WHICH" = box1 ] && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r104; mkdir -p $R/logs $R/results $R/ckpt || exit 1
BPE=data/pg19/bpe_8192.json
CM="--data-books data/pg19 --bpe $BPE --hc on --task bpe_recall --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500 --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6 --key-layer 0 --key-pos-free --freeze-query --query-init identity --n-steps 0 --beta 32 --memory-size 512 --batch 32 --store-grow --steps 10000 --shared-per-layer --shared-pl-key token"
PROG=$R/logs/_progress_fixed_${WHICH}.log
echo "r104-fixed $WHICH (detach) starting $(date -Is)" | tee $PROG
run () { name=$1; shift
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $CM "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -iE "recall" $R/logs/$name.log | tail -1 >> $PROG
}
# (1) r103 re-verify: n_seg 4, 32 facts, matching r103's spl_token seed
run r103_spl_token_detach_s${SEED} --n-seg 4 --n-facts 32 --seed $SEED
# (2) r104 capacity arm: n_seg 8, NF facts (the arm that OOM'd before the fix)
run r104_spl_token_f${NF} --n-seg 8 --n-facts $NF --seed 0
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -20 $PROG
