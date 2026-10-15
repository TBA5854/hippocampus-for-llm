#!/usr/bin/env bash
# r69 — diagnose the r68 multi-hop negative. Usage: run.sh box1|box2
set -u
export PATH="$HOME/.local/bin:$PATH"
WHICH="${1:?usage: run.sh box1|box2}"
case "$WHICH" in
  box1) ROOT="$HOME/research/framework"; INIT=experiment/r66/ckpt/r66_mlcurr_f16.pt ;;
  box2) ROOT="$HOME/fw";                 INIT=experiment/r66/ckpt/r66_mlcold_f16.pt ;;
  *) echo "usage: run.sh box1|box2" >&2; exit 1 ;;
esac
cd "$ROOT" || { echo "FATAL cd $ROOT" >&2; exit 1; }
R=experiment/r69
mkdir -p $R/logs $R/results $R/ckpt || { echo "FATAL: $ROOT not writable" >&2; exit 1; }
BPE=data/pg19/bpe_8192.json
BASE="--task bpe_hop --data-books data/pg19 --bpe $BPE --hc on --answer-weight 5.0
      --beta-grad 4.0 --beta-max 64.0 --d-mem 256 --d-model 256 --dg-bypass --eval-every 500
      --lr 0.0003 --n-heads 1 --n-layers 6 --readout graded --seg-len 128 --write-threshold 0.6
      --key-layer 0 --key-pos-free --freeze-query --query-init identity
      --n-steps 0 --beta 32 --value-skip-layer 0 --value-skip-norm
      --memory-size 512 --n-seg 4 --batch 32 --n-facts 16 --seed 0 --recency-read"
ML="--read-layers 0 1 2 3 4 5 --write-layers 0 1 2 3 4 5"
PROG=$R/logs/_progress_${WHICH}.log
echo "r69 $WHICH starting in $ROOT at $(date -Is)" | tee $PROG
run () { name=$1; shift
  echo "=== $name START $(date -Is)" >> $PROG
  uv run --group sleep python -u -m mech.train $BASE $ML "$@" \
    --out $R/results/$name.json --save $R/ckpt/$name.pt > $R/logs/$name.log 2>&1
  echo "=== $name EXIT $? $(date -Is)" >> $PROG
}
if [ "$WHICH" = "box2" ]; then
  # is 1-hop (base recall in hop format) learnable? then curriculum to 2-hop
  run r69_hop1_retrieved --n-hops 1 --hop-key retrieved --steps 4000 --init $INIT
  run r69_hop2_curr      --n-hops 2 --hop-key retrieved --steps 6000 \
      --init experiment/r69/ckpt/r69_hop1_retrieved.pt
else
  # format sanity (token, 1-hop) + does long training alone crack 2-hop?
  run r69_hop1_token --n-hops 1 --hop-key token     --steps 4000 --init $INIT
  run r69_hop2_long  --n-hops 2 --hop-key retrieved --steps 12000 --init $INIT
fi
echo "=== ROUND DONE $(date -Is)" >> $PROG
tail -1 $PROG
