#!/usr/bin/env bash
# r109 (#2, several shared stores) — is the 128-fact frontier PER-STORE? Shard facts across
# N stores by key token (tok mod N); each shard then holds ~128/N distinct keys. If recall
# lifts from the 1-shard baseline toward the ~96% seen at 64 facts, the frontier is an
# addressing/per-store limit (the user's KV-cache framing) and sharding relieves it.
# Same read-all/write-mean setup as r104's baseline (which flatlined at 0.3% @128), + N shards.
# Fixed memory_size 8192/shard (no lapping). box2. usage: run.sh box2
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="$HOME/fw"; cd "$ROOT" || { echo "FATAL cd" >&2; exit 1; }
R=experiment/r109; mkdir -p $R/logs $R/results $R/ckpt || exit 1
BPE=data/pg19/bpe_8192.json
CM="--data-books data/pg19 --bpe $BPE --hc on --task bpe_recall --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500 --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6 --key-layer 0 --key-pos-free --freeze-query --query-init identity --n-steps 0 --beta 32 --memory-size 2048 --n-seg 8 --batch 8 --n-facts 128 --steps 10000 --seed 0 --read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5 --value-skip-layer 0 --value-skip-norm"
PROG=$R/logs/_progress.log
echo "r109 (several stores: 128 facts, N shards) start $(date -Is)" | tee $PROG
run () { name=$1; shift
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $CM "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -iE "recall" $R/logs/$name.log | tail -1 >> $PROG
}
run r109_shard1 --n-shards 1
run r109_shard2 --n-shards 2
run r109_shard4 --n-shards 4
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -12 $PROG
