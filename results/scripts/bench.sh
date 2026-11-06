#!/bin/bash
# usage: bench.sh pie|vllm <q36|q38> <shape> <label> [extra args...]
#   shape: cN  (throughput: 2N requests, concurrency N, 128-token prompt + header + chat frame, 128 out)
#          pfN (prefill_batch: N requests at once, 512 + header + frame, 1 out)
#          raw (8 requests one at a time, pf prompts, 256 out)
# PIE_PY selects the pie engine's python dir (default the main checkout's).
eng=$1; m=$2; shape=$3; label=$4; shift 4
case $m in
  q36) MODEL=Qwen/Qwen3.6-27B; PC=/root/prompts/q36-c167.json; PP=/root/prompts/q36-pf551.json ;;
  q38) MODEL=Qwen/Qwen3.8-27B; PC=/root/prompts/q38-c205.json; PP=/root/prompts/q38-pf589.json ;;
  *) MODEL=$m; PC=${PC:?}; PP=${PP:?} ;;
esac
RES=${RES:-/root/r2g/evidence/results/runs}
mkdir -p $RES
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
  env -u CLAUDE_CODE_MESSAGING_TOKEN PYTHONPATH=${PIE_PY:-$P/packages/python/server/python}:$P/packages/python/client/src \
  PIE_BENCH_INFERLET_DIR=$P/examples/text-completion-bench \
  timeout 2400 /root/venv-pie/bin/python pie_bench.py "${mode[@]}" --model $MODEL --warmup 2 \
    --pretokenized-prompts --report-timing --json-out $RES/$label.json "$@" > $RES/$label.log 2>&1
else
  env -u CLAUDE_CODE_MESSAGING_TOKEN timeout 2400 /root/venv-vllm/bin/python vllm_bench.py "${mode[@]}" --model $MODEL --warmup 2 \
    --report-timing --json-out $RES/$label.json "$@" > $RES/$label.log 2>&1
fi
echo EXIT=$? >> $RES/$label.log
kill $CLK 2>/dev/null
