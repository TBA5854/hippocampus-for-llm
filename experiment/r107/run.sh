#!/usr/bin/env bash
# r107 — MAP the per-layer capacity frontier between 64 facts (~96%, r103/r104) and 128 (45.3%, r104).
# Per-layer write, cold, matched to r104 except n_facts. Fills in the decline: is it a cliff near
# ~100 or a gradual slope? box1 = 96 facts, box2 = 112 facts. batch 16 (< f128, fits).
# usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework"; NF=96;  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ;;
  box2) ROOT="$HOME/fw";                 NF=112 ;;   # no expandable_segments on box2 (NVML)
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r107; mkdir -p $R/logs $R/results $R/ckpt || exit 1
BPE=data/pg19/bpe_8192.json
CM="--data-books data/pg19 --bpe $BPE --hc on --task bpe_recall --answer-weight 5.0 --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500 --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6 --key-layer 0 --key-pos-free --freeze-query --query-init identity --n-steps 0 --beta 32 --memory-size 512 --store-grow --steps 10000 --shared-per-layer --shared-pl-key token --n-seg 8 --batch 16 --seed 0"
PROG=$R/logs/_progress_${WHICH}.log
echo "r107 $WHICH (NF=$NF) starting $(date -Is)" | tee $PROG
name=r107_spl_f${NF}
echo "=== $name START $(date -Is)" >> $PROG
uv run --group sleep python -u -m mech.train $CM --n-facts $NF \
  --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
echo "=== $name EXIT $? $(date -Is)" >> $PROG
grep -iE "recall" $R/logs/$name.log | tail -1 >> $PROG
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -6 $PROG
