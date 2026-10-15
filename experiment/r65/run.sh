#!/usr/bin/env bash
# r65 — CSA2 axis 3: how few bits does the store survive? (FP4 KV). OFFLINE quant probe, r50_bpe64.
# box2=main (bit sweep x pool none/token, at scale). box1=ablate (keys-only vs values-only).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r65
mkdir -p $R/logs $R/results || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
SFX="${SFX:-}"
UV="uv run --group sleep python -u -m mech.persist"
RUN=experiment/r50/results/r50_bpe64.json
CK=experiment/r50/ckpt/r50_bpe64.pt
BITS="16 8 6 4 3 2"
PROG=$R/logs/_progress${SFX}.log
echo "r65 starting in $ROOT at $(date -Is)" | tee $PROG
case "${1:?usage: run.sh main|ablate}" in
  main)   # bit sweep x {dense, token pool} at 15/31/63 windows
    {
      echo "=== r65 main (bits x pool) START $(date -Is)"
      for cfg in "2048 16" "4096 32" "8192 64"; do set -- $cfg
        echo "--- M=$1  bridge=$(( $2 - 1 )) windows ---"
        $UV quant --run $RUN --ckpt $CK --iters 10 --batch 16 \
            --mem-list $1 --nseg-list $2 --pool-modes none token \
            --quant-bits $BITS --quant-what both \
            --out $R/results/r65_both_M$1_n$2.json
      done
      echo "=== ROUND DONE $(date -Is)"
    } > $R/logs/r65_main.log 2>&1
    tail -1 $R/logs/r65_main.log
    ;;
  ablate) # keys-only vs values-only: which is bit-limiting? (dense, 31 windows)
    # box1 has r50_bpe16_s2 (r50_bpe64 lives on box2); also a 2nd-model cross-check.
    RUN=experiment/r50/results/r50_bpe16_s2.json
    CK=experiment/r50/ckpt/r50_bpe16_s2.pt
    {
      echo "=== r65 ablate (keys vs values, r50_bpe16_s2) START $(date -Is)"
      for w in keys values; do
        echo "--- quant $w ---"
        $UV quant --run $RUN --ckpt $CK --iters 10 --batch 16 \
            --mem-list 4096 --nseg-list 32 --pool-modes none \
            --quant-bits $BITS --quant-what $w \
            --out $R/results/r65_${w}_M4096_n32.json
      done
      echo "=== ROUND DONE $(date -Is)"
    } > $R/logs/r65_ablate.log 2>&1
    tail -1 $R/logs/r65_ablate.log
    ;;
esac
