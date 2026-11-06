#!/bin/bash
# usage: pie-pb.sh <model> <n> <outdir> <label> [extra pie_bench args...]
# pfN-512: N cold 512-token prompts offered at once, 1 output token, 16 warmup prompts first.
model=$1; n=$2; out=$3; label=$4; shift 4
mkdir -p $out
cd /root/r2g/pie/scripts/bench
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader > $out/$label.clocks
env -u CLAUDE_CODE_MESSAGING_TOKEN \
PYTHONPATH=${PIE_PY:-/root/r2g/pie/packages/python/server/python}:/root/r2g/pie/packages/python/client/src \
PIE_BENCH_INFERLET_DIR=/root/r2g/pie/examples/text-completion-bench \
/root/venv-pie/bin/python pie_bench.py tput --model $model \
  --num-requests $n --concurrency $n --warmup 16 --max-tokens 1 \
  --prompts-file /root/prompts-gptoss-512.json \
  --json-out $out/$label.json "$@" > $out/$label.log 2>&1
echo EXIT=$? >> $out/$label.log
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader >> $out/$label.clocks
