#!/usr/bin/env bash
# r92 — fine-key, done right. r89 (warm from single-token r79) collapsed = confound; r91
# (cold collision) didn't bootstrap. Here: bootstrap the read path on the EASY recall task
# WITH the bigram key from step 0, then WARM the collision from it — a compatible warm start.
#   box1 = fine arm: recall+fine-key cold (crux: does the bigram bootstrap?) -> collision+fine warm
#   box2 = exact arm (matched baseline): recall exact cold -> collision exact warm
# usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework" ;;
  box2) ROOT="$HOME/fw" ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r92
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
COMMON="--data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0 --beta-grad 4.0
      --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500 --lr 0.0003
      --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity --n-steps 0 --beta 32
      --memory-size 512 --n-seg 4 --batch 32 --n-facts 32 --read-layers 0 1 2 3 4 5
      --write-layers 0 1 2 3 4 5 --value-skip-layer 0 --value-skip-norm"
PROG=$R/logs/_progress_${WHICH}.log
echo "r92 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
if [ "$WHICH" = "box1" ]; then
  # FINE arm: bootstrap the bigram read path on recall, then warm collision from it
  run r92_recall_fine_cold  --task bpe_recall    --fine-key --steps 12000 --seed 0
  run r92_col_C4_fine_warm  --task bpe_collision --fine-key --collisions-per-key 4 --steps 6000 --seed 0 \
                            --init $R/ckpt/r92_recall_fine_cold.pt
else
  # EXACT arm (matched baseline): recall exact cold, then collision exact warm
  run r92_recall_exact_cold --task bpe_recall    --steps 12000 --seed 0
  run r92_col_C4_exact_warm --task bpe_collision --collisions-per-key 4 --steps 6000 --seed 0 \
                            --init $R/ckpt/r92_recall_exact_cold.pt
fi
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG