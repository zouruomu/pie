#!/bin/bash
# usage: prof-pie.sh <q36|q38> <cN> <label> [extra pie_bench args...]   (PIE_PY selects the engine)
# nsys over the whole measured window (cudaProfilerStart/Stop), graph nodes traced; traces stay in /root/prof.
m=$1; shape=$2; label=$3; shift 3
case $m in q36) MODEL=Qwen/Qwen3.6-27B ;; q38) MODEL=Qwen/Qwen3.8-27B ;; esac
n=${shape#c}; OUT=/root/r2g/evidence/results/prof; mkdir -p $OUT
P=/root/r2g/pie
cd $P/scripts/bench
env -u CLAUDE_CODE_MESSAGING_TOKEN PYTHONPATH=${PIE_PY:-$P/packages/python/server/python}:$P/packages/python/client/src \
PIE_BENCH_INFERLET_DIR=$P/examples/text-completion-bench \
/usr/local/cuda/bin/nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o /root/prof/$label \
/root/venv-pie/bin/python pie_bench.py tput --model $MODEL --num-requests $((2*n)) --concurrency $n --max-tokens 128 \
  --prompts-file /root/prompts/site-c128.json --warmup 2 --pretokenized-prompts --cuda-profiler-capture \
  --json-out $OUT/$label.json "$@" > $OUT/$label.log 2>&1
echo EXIT=$? >> $OUT/$label.log
/usr/local/cuda/bin/nsys export -t sqlite -f true -o /root/prof/$label.sqlite /root/prof/$label.nsys-rep > /dev/null 2>&1
