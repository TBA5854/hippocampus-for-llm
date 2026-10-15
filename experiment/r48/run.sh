#!/usr/bin/env bash
# r48 — BPE toward the ceiling (curriculum) and past the char 13-fact cap (scaling).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r48
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
echo "r48 starting in $ROOT at $(date -Is)"
BPE=data/pg19/bpe_8192.json
BASE="--task bpe_recall --data-books data/pg19 --bpe $BPE --hc on
      --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256
      --dg-bypass --eval-every 500 --lr 0.0003 --n-heads 1 --n-layers 6
      --readout graded --seg-len 128 --seed 1 --write-threshold 0.6 --key-layer 0
      --read-layers 0 --key-pos-free --freeze-query --query-init identity --n-steps 0
      --beta 32 --write-layer 5 --value-skip-layer 0 --value-skip-norm
      --memory-size 512 --n-seg 4 --batch 32"
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}
case "${1:?usage: run.sh ladder|scale}" in
  ladder)  # does the curriculum take BPE 8 facts to the 96% ceiling?
    run r48_bpe4      --n-facts 4  --steps 6000
    run r48_bpe8_curr --n-facts 8  --steps 12000 --init experiment/r48/ckpt/r48_bpe4.pt
    ;;
  scale)   # past the char 13-fact cap: 16 and 32 facts cold
    run r48_bpe16     --n-facts 16 --steps 6000
    run r48_bpe32     --n-facts 32 --steps 6000
    ;;
esac
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
