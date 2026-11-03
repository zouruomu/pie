#!/bin/bash
# usage: bench-model.sh <hf-model> <knobfile|none> <label> [extra pie_bench args...]
source /etc/pod/env.sh
model=$1; k=$2; label=$3; shift 3
u=/root/.pie/profiles/cuda/user/nvidia-geforce-rtx-4090.toml
rm -f $u; [ "$k" != none ] && cp /root/knobs/$k.toml $u
mkdir -p /root/results/xm
cd /root/src/pie/scripts/bench
PYTHONPATH=/root/src/pie/packages/python/client/src:/root/src/pie/packages/python/server/python \
PIE_BENCH_INFERLET_DIR=/root/src/pie/examples/text-completion-bench \
/root/venv-pie/bin/python pie_bench.py latency --model $model \
  --requests 8 --warmup 2 --max-tokens 256 --prompts-file /root/prompts-512.json \
  --report-timing --dump-all-token-ids --json-out /root/results/xm/$label.json "$@" > /root/results/xm/$label.log 2>&1
echo EXIT=$? >> /root/results/xm/$label.log
rm -f $u
