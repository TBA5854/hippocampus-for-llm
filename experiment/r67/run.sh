#!/usr/bin/env bash
# r67 — update / recency. Does a recency-biased read return the NEWEST value of a
# re-stated fact? Warm from r66_mlcurr_f16 (ml placement, 100% @16). box1.
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r67
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
INIT=experiment/r66/ckpt/r66_mlcurr_f16.pt
BASE="--task bpe_update --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --n-steps 0 --beta 32 --value-skip-layer 0 --value-skip-norm
      --memory-size 512 --n-seg 4 --batch 32 --n-facts 16 --seed 0"
ML="--read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5"
PROG=$R/logs/_progress.log
echo "r67 starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $BASE $ML "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
# baseline: no recency (also tests "position-in-key gives recency for free")
run r67_recency_off --steps 6000 --init $INIT
# the mechanism: recency-biased read
run r67_recency_on  --steps 6000 --recency-read --init $INIT
# control: recency must not disturb ordinary recall of a non-updated fact
run r67_nonupd      --steps 4000 --recency-read --query-nonupdated --init $INIT
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
