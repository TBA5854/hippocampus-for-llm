#!/usr/bin/env bash
# r106 (revised) — close the 128-fact gap with FASTER, higher-value levers. L12 dropped: 12
# per-layer reads x accum 4 ran ~10h with no eval in 31 min, intractable. Instead:
#   curr32 = CURRICULUM: warm 128 facts from the r103 32-fact per-layer ckpt. The curriculum
#            broke every earlier fact-count cliff (r25/r49/r66); does it close 128? L6, batch 16
#            (= r104's config), so fast (~2.5h).
#   d512   = WIDTH: d_model/d_mem 512 (vs 256), cold, per-layer 128. batch 4 accum 4 (wider->low batch).
# Baseline to beat = r104 spl_token_f128 (cold) = 45.3%. box1 only. usage: run.sh box1
set -u
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # box1 driver OK (never box2)
ROOT="$HOME/research/framework"; cd "$ROOT" || { echo "FATAL cd" >&2; exit 1; }
R=experiment/r106; mkdir -p $R/logs $R/results $R/ckpt || exit 1
BPE=data/pg19/bpe_8192.json
CM="--data-books data/pg19 --bpe $BPE --hc on --task bpe_recall --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0 --dg-bypass --eval-every 500 --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6 --key-layer 0 --key-pos-free --freeze-query --query-init identity --n-steps 0 --beta 32 --memory-size 512 --store-grow --steps 10000 --shared-per-layer --shared-pl-key token --n-seg 8 --n-facts 128 --seed 0"
PROG=$R/logs/_progress_box1.log
echo "r106 (128-fact: curriculum + width) box1 starting $(date -Is)" | tee $PROG
run () { name=$1; shift
  [ -f "$R/results/$name.json" ] && { echo "SKIP $name" >> $PROG; return; }
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $CM "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -iE "recall" $R/logs/$name.log | tail -1 >> $PROG
}
# curriculum: warm 128 from the 32-fact per-layer ckpt (fast, L6 batch 16)
run r106_spl_f128_curr32 --batch 16 --init experiment/r103/ckpt/r103_spl_token.pt
# width: d512 cold (batch 4 accum 4 for memory)
run r106_spl_f128_d512   --batch 4 --accum 4 --d-model 512 --d-mem 512
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -12 $PROG
