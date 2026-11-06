#!/bin/bash
# nsys of the final build at c32 on cut A: which selected-attention instantiation a 32-row decode runs.
M=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/mini60-e16
cd /root/r2g/pie/scripts/bench
env -u CLAUDE_CODE_MESSAGING_TOKEN PYTHONPATH=/root/r2g/pie/packages/python/client/src:/root/r2g/pie/packages/python/server/python \
PIE_BENCH_INFERLET_DIR=/root/r2g/pie/examples/text-completion-bench \
/usr/local/cuda/bin/nsys profile -t cuda -s none --cpuctxsw=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o /root/prof/final-a-c32 \
/root/venv-pie/bin/python pie_bench.py tput --model $M --prompts-file /root/prompts-256-a.json --max-tokens 128 \
  --concurrency 32 --num-requests 64 --warmup 2 --report-timing --json-out /root/prof/final-a-c32.json \
  --cuda-profiler-capture --cuda-profiler-delay-s 1.5 --cuda-profiler-duration-s 0.5 > /root/prof/final-a-c32.log 2>&1
echo EXIT=$? >> /root/prof/final-a-c32.log
