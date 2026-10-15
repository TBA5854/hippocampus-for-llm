#!/usr/bin/env bash
# r99 — "multilayer CoT does semantics": a CoT whose FIRST hop is served by PARAMETRIC memory
# (a fixed syn->canon lexicon learned in weights, NOT stored) and second by the store
# (canon->val). Query the synonym; if hop1 (syn->canon) is learned and hop2 (canon->val) reads,
# the model does semantic recall by reasoning, though the store itself can't fuzzy-match (r98).
# broken = random syn->canon per example (unlearnable) -> hop1 must fail (control).
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
R=experiment/r99
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
BASE=experiment/r79/ckpt/r79_shared_gate_recall.pt
COMMON="--task bpe_semchain --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity --n-steps 0 --beta 32
      --memory-size 1024 --n-seg 4 --batch 32 --n-facts 8 --read-layers 0 1 2 3 4 5
      --write-layers 0 1 2 3 4 5 --value-skip-layer 0 --value-skip-norm --recency-read
      --write-op-gate --write-skip-pad --init $BASE"
PROG=$R/logs/_progress_${WHICH}.log
echo "r99 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $COMMON "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
  grep -E "^step +10000|^step +8000" $R/logs/$name.log | tail -1 >> $PROG
}
if [ "$WHICH" = "box1" ]; then
  run r99_semchain        --lexicon-size 64 --steps 10000 --seed 0
  run r99_semchain_broken --lexicon-size 64 --steps 4000  --seed 0 --broken-chain
else
  run r99_semchain_s1     --lexicon-size 64 --steps 10000 --seed 1
  run r99_semchain_L32    --lexicon-size 32 --steps 10000 --seed 0
fi
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG