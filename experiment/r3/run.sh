#!/usr/bin/env bash
# r3 — same-space keys (layer-1 key, identity query init) against the fact-count cliff.
# SEQUENTIAL: r2 measured 5 parallel runs ~15% slower per run than one at a time,
# so parallelism only delays the most decisive answer.
set -u
export PATH="$HOME/.local/bin:$PATH"      # detached shells are non-login; uv lives here
cd ~/research/framework
R=experiment/r3
mkdir -p $R/logs $R/results $R/ckpt

BASE="--task mqar --hc on --answer-weight 5.0 --batch 32 --beta-grad 4.0 --beta-max 64.0
      --d-mem 256 --d-model 256 --data data/tinyshakespeare.txt --dg-bypass --eval-every 250
      --lr 0.0003 --memory-size 512 --n-heads 1 --n-layers 6 --n-seg 4 --read-layers 1
      --readout graded --seed 0 --seg-len 128 --steps 4000 --write-threshold 0.6"
SAME="--write-layer 5 --key-layer 1 --value-shift 0 --query-init identity"

run() {
  name=$1; shift
  echo "=== $name START $(date -Is)" >> $R/logs/_progress.log
  uv run --group sleep python -u -m mech.train $BASE "$@" \
      --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $R/logs/_progress.log
}

run r3_f8         $SAME --n-facts 8
run r3_f8_shuf    $SAME --n-facts 8 --shuffle-store
run r3_f2         $SAME --n-facts 2
run r3_f16        $SAME --n-facts 16
run r3_w5s1_align --write-layer 5 --value-shift 1 --n-facts 2
# r2_w4 (the 100% reference) also dropped to 0% identity-slot reads while its
# recall rose, so the instrument check has to be made at the offset it really
# uses: if the diagnostic is right, this model reads at +1.
run r3_w4_align   --write-layer 4 --value-shift 1 --n-facts 2
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
