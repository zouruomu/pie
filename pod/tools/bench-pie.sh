#!/bin/bash
# usage: bench-pie.sh <label> [extra pie_bench args...]
source /etc/pod/env.sh
label=$1; shift
mkdir -p /root/results
cd /root/src/pie/scripts/bench
PYTHONPATH=/root/src/pie/packages/python/client/src:/root/src/pie/packages/python/server/python \
PIE_BENCH_INFERLET_DIR=/root/src/pie/examples/text-completion-bench \
/root/venv-pie/bin/python pie_bench.py latency --model Qwen/Qwen3.5-0.8B \
  --requests 8 --warmup 2 --max-tokens 256 --prompts-file /root/prompts-512.json \
  --report-timing --json-out /root/results/$label.json "$@" > /root/results/$label.log 2>&1
echo EXIT=$? >> /root/results/$label.log
