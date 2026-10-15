#!/usr/bin/env bash
# r110 (box1, chained after r108) — does SHARDING SCALE capacity? r109 tests 1/2/4 shards @128;
# this tests whether more shards hold MORE facts: 2/4/8 shards @256 facts (single store saturates
# ~45% @128, so 256 in one store is hopeless; if sharding N-fold relieves it, 8-shard @256 = ~32
# facts/shard -> should be recoverable). Same read-all/write-mean + n_shards as r109.
set -u
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # box1 driver OK
ROOT="$HOME/research/framework"; cd "$ROOT" || { echo "FATAL cd" >&2; exit 1; }
R=experiment/r110; mkdir -p $R/logs $R/results $R/ckpt || exit 1
BPE=data/pg19/bpe_8192.json
CM="--data-books data/pg19 --bpe $BPE --hc on --task bpe_recall --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500 --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6 --key-layer 0 --key-pos-free --freeze-query --query-init identity --n-steps 0 --beta 32 --memory-size 4096 --n-seg 16 --batch 8 --n-facts 256 --steps 10000 --seed 0 --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5 --value-skip-layer 0 --value-skip-norm"
PROG=$R/logs/_progress.log
echo "r110 (sharding capacity scaling @256 facts) start $(date -Is)" | tee $PROG
run () { name=$1; shift
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $CM "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -iE "recall" $R/logs/$name.log | tail -1 >> $PROG
}
run r110_shard2_f256 --n-shards 2
run r110_shard4_f256 --n-shards 4
run r110_shard8_f256 --n-shards 8
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -12 $PROG
