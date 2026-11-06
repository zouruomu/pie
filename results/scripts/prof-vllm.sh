#!/bin/bash
# usage: prof-vllm.sh <label> <graph-trace: node|graph> [extra args]; in-process engine core so the bench's
# cudaProfilerStart/Stop window covers it. Traces stay in /root/prof.
label=$1; gt=$2; shift 2
M=${MODEL:-/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/mini60-e16}
cd /root/r2g/pie/scripts/bench
env -u CLAUDE_CODE_MESSAGING_TOKEN VLLM_ENABLE_V1_MULTIPROCESSING=0 \
/usr/local/cuda/bin/nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=$gt \
  --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o /root/prof/$label \
${VENV:-/root/venv-vllm}/bin/python vllm_bench.py latency --model $M --requests 2 --warmup 2 --max-tokens 256 \
  --prompts-file ${PROMPTS:-/root/prompts-512.json} --report-timing --json-out /root/prof/$label.json \
  --cuda-profiler-capture "$@" > /root/prof/$label.log 2>&1
echo EXIT=$? >> /root/prof/$label.log
