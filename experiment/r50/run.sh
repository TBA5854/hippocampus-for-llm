#!/usr/bin/env bash
# r50 — replicate the 100% ladder on seed 2 (box1) + push to 64 facts (box2).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/research/framework}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r50
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
echo "r50 starting in $ROOT at $(date -Is)"
BPE=data/pg19/bpe_8192.json
B="--task bpe_recall --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
   --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
   --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
   --key-layer 0 --read-layers 0 --key-pos-free --freeze-query --query-init identity
   --n-steps 0 --beta 32 --write-layer 5 --value-skip-layer 0 --value-skip-norm
   --memory-size 512 --n-seg 4 --batch 32"
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $B "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}
case "${1:?usage: run.sh seed2|f64}" in
  seed2)  # replicate the ladder on a fresh seed: 4 cold -> 8 warm -> 16 warm
    run r50_bpe4_s2  --seed 2 --n-facts 4  --steps 6000
    run r50_bpe8_s2  --seed 2 --n-facts 8  --steps 12000 --init experiment/r50/ckpt/r50_bpe4_s2.pt
    run r50_bpe16_s2 --seed 2 --n-facts 16 --steps 12000 --init experiment/r50/ckpt/r50_bpe8_s2.pt
    ;;
  f64)    # push scaling: 64 facts warm from the 32-fact 100% model; does the 512-slot store bind?
    run r50_bpe64 --seed 1 --n-facts 64 --steps 12000 --init experiment/r49/ckpt/r49_bpe32_curr.pt
    ;;
esac
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
