#!/usr/bin/env bash
# r64 — CSA2 candidate-pool read: shared store + reuse. OFFLINE pool probe.
# box2=main (pool ablation at scale, r50_bpe64). box1=reuse (cross-layer reuse, MQAR read@all).
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="${ROOT:-$HOME/fw}"
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r64
mkdir -p $R/logs $R/results || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
SFX="${SFX:-}"
UV="uv run --group sleep python -u -m mech.persist"
PROG=$R/logs/_progress${SFX}.log
echo "r64 starting in $ROOT at $(date -Is)" | tee $PROG
case "${1:?usage: run.sh main|reuse}" in
  main)
    RUN=experiment/r50/results/r50_bpe64.json; CK=experiment/r50/ckpt/r50_bpe64.pt
    {
      echo "=== r64 main pool ablation (none/token/ann) START $(date -Is)"
      for cfg in "2048 16" "4096 32" "8192 64"; do set -- $cfg
        echo "--- M=$1  bridge=$(( $2 - 1 )) windows ---"
        $UV pool --run $RUN --ckpt $CK --iters 10 --batch 16 \
            --mem-list $1 --nseg-list $2 --pool-modes none token ann --pool-size 16 \
            --out $R/results/r64_bpe64_M$1_n$2.json
      done
      echo "=== ROUND DONE $(date -Is)"
    } > $R/logs/r64_main.log 2>&1
    tail -1 $R/logs/r64_main.log
    ;;
  reuse)
    # cross-layer reuse on a read@all model: build pool once at anchor vs per-layer
    RUN=experiment/r60/results/r60_sh_f16_s2.json; CK=experiment/r60/ckpt/r60_sh_f16_s2.pt
    {
      echo "=== r64 reuse (per-layer vs anchor) START $(date -Is)"
      echo "--- per-layer rebuild (anchor none) ---"
      $UV pool --run $RUN --ckpt $CK --iters 10 --batch 16 \
          --mem-list 512 --nseg-list 4 --pool-modes none token ann --pool-size 16 \
          --out $R/results/r64_reuse_perlayer.json
      echo "--- cross-layer reuse (anchor 0) ---"
      $UV pool --run $RUN --ckpt $CK --iters 10 --batch 16 --pool-anchor 0 \
          --mem-list 512 --nseg-list 4 --pool-modes none token ann --pool-size 16 \
          --out $R/results/r64_reuse_anchor0.json
      echo "=== ROUND DONE $(date -Is)"
    } > $R/logs/r64_reuse.log 2>&1
    tail -1 $R/logs/r64_reuse.log
    ;;
esac
