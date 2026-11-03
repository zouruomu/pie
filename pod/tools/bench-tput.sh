#!/bin/bash
# usage: bench-tput.sh <hf-model> <knobfile|none> <label> <concurrency> <num-requests> [extra...]
source /etc/pod/env.sh
model=$1; k=$2; label=$3; conc=$4; num=$5; shift 5
u=/root/.pie/profiles/cuda/user/nvidia-geforce-rtx-4090.toml
rm -f $u; [ "$k" != none ] && cp /root/knobs/$k.toml $u
mkdir -p /root/results/xm
cd /root/src/pie/scripts/bench
PYTHONPATH=/root/src/pie/packages/python/client/src:/root/src/pie/packages/python/server/python \
PIE_BENCH_INFERLET_DIR=/root/src/pie/examples/text-completion-bench \
/root/venv-pie/bin/python pie_bench.py tput --model $model \
  --num-requests $num --concurrency $conc --warmup 2 --max-tokens 256 \
  --json-out /root/results/xm/$label.json "$@" > /root/results/xm/$label.log 2>&1
echo EXIT=$? >> /root/results/xm/$label.log
rm -f $u
