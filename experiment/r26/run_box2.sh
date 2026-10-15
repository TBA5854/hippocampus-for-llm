#!/usr/bin/env bash
# r26 (box2) — decode probe on the 48-fact 1x checkpoint, which box2 trained in r25.
# Denser in fact symbols than the 8x store, so it is the harder case for H26c.
set -u
cd ~/fw
PY=~/research/framework/.venv/bin/python
R=experiment/r26
mkdir -p $R/logs $R/results

echo "=== r26_decode_1x_f48 START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
$PY -u -m mech.decode \
    --run experiment/r25/results/r25_1x_f48_warm.json \
    --ckpt experiment/r25/ckpt/r25_1x_f48_warm.pt \
    --batch 8 --out $R/results/r26_decode_1x_f48.json > $R/logs/r26_decode_1x_f48_box2.log 2>&1
echo "=== r26_decode_1x_f48 EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log

echo "=== r26_decode_1x_f8 START $(date -Is) [box2]" >> $R/logs/_progress_box2.log
$PY -u -m mech.decode \
    --run experiment/r23/results/r23_1x_sym128_f8.json \
    --ckpt experiment/r23/ckpt/r23_1x_sym128_f8.pt \
    --batch 8 --out $R/results/r26_decode_1x_f8.json > $R/logs/r26_decode_1x_f8_box2.log 2>&1
echo "=== r26_decode_1x_f8 EXIT $? $(date -Is) [box2]" >> $R/logs/_progress_box2.log
echo "=== ROUND DONE $(date -Is) [box2]" >> $R/logs/_progress_box2.log
