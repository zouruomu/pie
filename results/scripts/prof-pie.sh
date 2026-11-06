#!/bin/bash
# usage: prof-pie.sh <label> <graph-trace: node|graph> [extra pie_bench args...]; traces stay in /root/prof
label=$1; gt=$2; shift 2
M=${MODEL:-/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/mini60-e16}
cd /root/r2g/pie/scripts/bench
env -u CLAUDE_CODE_MESSAGING_TOKEN PYTHONPATH=/root/r2g/pie/packages/python/client/src:/root/r2g/pie/packages/python/server/python \
PIE_BENCH_INFERLET_DIR=/root/r2g/pie/examples/text-completion-bench \
/usr/local/cuda/bin/nsys profile -t cuda,nvtx,osrt -s none --cpuctxsw=none --cuda-graph-trace=$gt \
  --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o /root/prof/$label \
/root/venv-pie/bin/python pie_bench.py latency --model $M \
  --requests 2 --warmup 2 --max-tokens 256 --prompts-file ${PROMPTS:-/root/prompts-512.json} \
  --report-timing --json-out /root/prof/$label.json \
  --cuda-profiler-capture --cuda-profiler-delay-s 1.0 --cuda-profiler-duration-s 0.5 "$@" > /root/prof/$label.log 2>&1
echo EXIT=$? >> /root/prof/$label.log
