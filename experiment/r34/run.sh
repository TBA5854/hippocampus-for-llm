#!/usr/bin/env bash
# r34 — PG-19. Four runs, matched on everything but the memory.
#
# The headline metric is the PER-SEGMENT loss profile, already printed each eval
# as [s0:.. s1:.. s2:.. s3:..]. forward() carries no KV cache, so for a
# memory-off model the segment index means nothing and the profile must be flat;
# any downward slope in the memory-on model is the store and cannot be bought
# with parameters.
#
# Store sizing: seg_len 256 writes 255 slots/segment, 4 segments = 1020 slots
# into 1024 — full, but no overflow, so the slope is not confounded by eviction.
#
#   box1: ROOT=~/research/framework   box2: ROOT=~/fw
set -eu
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT"
UV="$HOME/.local/bin/uv"
mkdir -p experiment/r34/logs experiment/r34/results

COMMON="--task lm --data-books data/pg19 --n-seg 4 --seg-len 256 --batch 24 \
        --steps 10000 --eval-every 250 --seed 0"

# The placement every working run in this project uses, and the reason it works:
# the key is a POSITION-FREE LAYER-0 state, i.e. the token embedding itself, and
# the query is a FROZEN IDENTITY projection of the same layer-0 state. Query and
# key therefore live in one space and matching is near-exact token lookup (r9;
# r31d showed that destroying this exactness with a projection puts recall at
# chance). Write stays at the last block, which is the objective's architecture.
#
# The first pass of this round used `--read-layers 1 --write-layer 5` with none
# of these flags, so the query was a random projection of a layer-1 state matched
# against last-block keys -- two different spaces, a configuration that has never
# worked even on the synthetic task. It produced a clean null that would have
# been reported as "memory does not help on natural text". It was measuring the
# configuration, not the corpus.
MEM="--hc on --read-layers 0 --key-layer 0 --key-pos-free --query-init identity \
     --freeze-query --write-layer 5 --memory-size 1024 --beta-grad 4 --dg-bypass"

launch () {  # launch <tag> <args...>
  tag="$1"; shift
  setsid nohup $UV run --group sleep python -u -m mech.train $COMMON "$@" \
    --out experiment/r34/results/${tag}.json \
    > experiment/r34/logs/${tag}.log 2>&1 < /dev/null &
  echo "launched $tag pid $!"
}

case "${1:-all}" in
  pair)                                   # the core comparison (box2)
    launch r34_off      --hc off
    launch r34_mem      $MEM
    ;;
  mem)                                    # memory arm only: r34_off is unaffected
    launch r34_mem      $MEM               # by the placement fix and is kept
    ;;
  seed1)                                  # replication: the effect is ~0.01 nats,
    launch r34_off_s1   --hc off --seed 1  # which one seed cannot carry
    launch r34_mem_s1   $MEM --seed 1
    ;;
  seed)                                   # `run.sh seed N` — a matched pair at
    n="${2:?usage: run.sh seed N}"         # seed N. Seeds 0 and 1 disagreed by 3x
    launch r34_off_s${n}  --hc off --seed "$n"
    launch r34_mem_s${n}  $MEM --seed "$n"
    ;;
  controls)                               # scramble + compression (box1)
    launch r34_shuffle  $MEM --shuffle-store
    launch r34_compress $MEM --memory-size 256 --write-select surprisal --write-frac 0.25
    ;;
esac
