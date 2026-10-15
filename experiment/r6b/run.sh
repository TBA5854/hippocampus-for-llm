#!/usr/bin/env bash
# r6b — r6_f8_M2048 OOM'd at batch 32 (15 GB held by one process: every segment's
# [B,S,M] settle intermediates live until backward, so memory scales with
# n_seg x M). Same run at batch 8; the batch change is the only difference.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd ~/research/framework
R=experiment/r6b
mkdir -p $R/logs $R/results $R/ckpt
echo "=== r6b_f8_M2048_b8 START $(date -Is)" >> $R/logs/_progress.log
uv run --group sleep python -u -m mech.train --task mqar --hc on --n-facts 8 \
    --answer-weight 5.0 --batch 8 --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 \
    --data data/tinyshakespeare.txt --dg-bypass --eval-every 250 --lr 0.0003 --n-heads 1 \
    --n-layers 6 --readout graded --seg-len 128 --steps 4000 --write-threshold 0.6 --seed 0 \
    --write-layer 5 --key-layer 0 --read-layers 0 --value-shift 0 --query-init identity \
    --memory-size 2048 --n-seg 16 \
    --out $R/results/r6b_f8_M2048_b8.json --save $R/ckpt/r6b_f8_M2048_b8.pt \
    > $R/logs/r6b_f8_M2048_b8.log 2>&1
echo "=== r6b_f8_M2048_b8 EXIT $? $(date -Is)" >> $R/logs/_progress.log
echo "=== ROUND DONE $(date -Is)" >> $R/logs/_progress.log
