#!/bin/bash
# usage: bench-comp.sh sglang|vllm <label> [extra args...]
source /etc/pod/env.sh
eng=$1; label=$2; shift 2
mkdir -p /root/results
cd /root/src/pie/scripts/bench
/root/venv-$eng/bin/python ${eng}_bench.py latency --model Qwen/Qwen3.5-0.8B \
  --requests 8 --warmup 2 --max-tokens 256 --prompts-file /root/prompts-512.json \
  --report-timing --json-out /root/results/$label.json "$@" > /root/results/$label.log 2>&1
echo EXIT=$? >> /root/results/$label.log
