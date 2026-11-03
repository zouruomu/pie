#!/bin/bash
# usage: prof-pie.sh <label> <graph-trace: node|graph> [extra pie_bench args...]
source /etc/pod/env.sh
label=$1; gt=$2; shift 2
mkdir -p /root/results/analysis
cd /root/src/pie/scripts/bench
PYTHONPATH=/root/src/pie/packages/python/client/src:/root/src/pie/packages/python/server/python \
PIE_BENCH_INFERLET_DIR=/root/src/pie/examples/text-completion-bench \
/usr/local/cuda/bin/nsys profile -t cuda,nvtx,osrt -s none --cpuctxsw=none --cuda-graph-trace=$gt \
  --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o /root/results/analysis/$label \
/root/venv-pie/bin/python pie_bench.py latency --model Qwen/Qwen3.5-0.8B \
  --requests 4 --warmup 2 --max-tokens 256 --prompts-file /root/prompts-512.json \
  --report-timing --json-out /root/results/analysis/$label.json \
  --cuda-profiler-capture --cuda-profiler-delay-s 0.70 --cuda-profiler-duration-s 0.30 "$@" > /root/results/analysis/$label.log 2>&1
echo EXIT=$? >> /root/results/analysis/$label.log
