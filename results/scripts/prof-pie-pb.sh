#!/bin/bash
# usage: prof-pie-pb.sh <model> <n> <outdir> <label> [extra pie_bench args...]
# nsys over the measured window of a pfN-512 run (node-level graph trace), then kernel totals by name.
model=$1; n=$2; out=$3; label=$4; shift 4
mkdir -p $out
cd /root/r2g/pie/scripts/bench
env -u CLAUDE_CODE_MESSAGING_TOKEN \
PYTHONPATH=${PIE_PY:-/root/r2g/pie/packages/python/server/python}:/root/r2g/pie/packages/python/client/src \
PIE_BENCH_INFERLET_DIR=/root/r2g/pie/examples/text-completion-bench \
/usr/local/cuda/bin/nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o $out/$label \
/root/venv-pie/bin/python pie_bench.py tput --model $model \
  --num-requests $n --concurrency $n --warmup 16 --max-tokens 1 \
  --prompts-file /root/prompts-gptoss-512.json --cuda-profiler-capture \
  --json-out $out/$label.json "$@" > $out/$label.log 2>&1
echo EXIT=$? >> $out/$label.log
/usr/local/cuda/bin/nsys stats -q --report cuda_gpu_kern_sum --format csv -o $out/$label $out/$label.nsys-rep > /dev/null 2>&1
