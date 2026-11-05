#!/bin/bash
# usage: bench.sh pie|vllm <shape> <label> [extra args...]
#   shape: cN  (throughput: 2N requests, concurrency N, 169-token prompts (128 + Gemma's chat frame), 128 out)
#          pfN (prefill_batch: N requests at once, 553-token prompts, 1 out)
#          raw (8 requests one at a time, 553-token prompts, 256 out)
# Results land in $RES (default /root/r2g/evidence/results/runs). MODEL defaults to Gemma-4-31B.
eng=$1; shape=$2; label=$3; shift 3
RES=${RES:-/root/r2g/evidence/results/runs}
MODEL=${MODEL:-google/gemma-4-31B-it}
PC=${PC:-/root/prompts/p169.json}; PP=${PP:-/root/prompts/p553.json}
mkdir -p $RES
# A run whose label matches the regex in /root/r2g/hold waits until the file is gone.
while [ -e /root/r2g/hold ] && [[ $label =~ $(cat /root/r2g/hold) ]]; do sleep 10; done
P=/root/r2g/pie
case $shape in
  c*)  n=${shape#c}; mode=(tput --num-requests $((2*n)) --concurrency $n --max-tokens 128 --prompts-file $PC) ;;
  pf*) n=${shape#pf}; mode=(tput --num-requests $n --concurrency $n --max-tokens 1 --prompts-file $PP) ;;
  raw) mode=(latency --requests 8 --max-tokens 256 --prompts-file $PP)
       [ $eng = vllm ] && mode=(tput --num-requests 8 --concurrency 1 --max-tokens 256 --prompts-file $PP) ;;
esac
cd $P/scripts/bench
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader -lms 1000 > $RES/$label.clk 2>/dev/null &
CLK=$!
if [ $eng = pie ]; then
  env -u CLAUDE_CODE_MESSAGING_TOKEN PYTHONPATH=$P/packages/python/client/src:$P/packages/python/server/python \
  PIE_BENCH_INFERLET_DIR=$P/examples/text-completion-bench \
  timeout 2400 /root/venv-pie/bin/python pie_bench.py "${mode[@]}" --model $MODEL --warmup 2 \
    --pretokenized-prompts --report-timing --json-out $RES/$label.json "$@" > $RES/$label.log 2>&1
else
  env -u CLAUDE_CODE_MESSAGING_TOKEN timeout 2400 /root/venv-vllm/bin/python vllm_bench.py "${mode[@]}" --model $MODEL --warmup 2 \
    --report-timing --json-out $RES/$label.json "$@" > $RES/$label.log 2>&1
fi
echo EXIT=$? >> $RES/$label.log
kill $CLK 2>/dev/null
