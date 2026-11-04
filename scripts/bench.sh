#!/bin/bash
# usage: bench.sh pie|vllm <shape> <label> [extra args...]
#   shape: cN (throughput: 2N requests, concurrency N, 225-token prompts, 128 out)
#          pfN (prefill_batch: N requests at once, 609-token prompts, 1 out)
#          raw (8 requests one at a time, 609-token prompts, 256 out)
# Results land in $RES (default /root/r2g/evidence/results/runs).
source /etc/pod/env.sh
eng=$1; shape=$2; label=$3; shift 3
RES=${RES:-/root/r2g/evidence/results/runs}
mkdir -p $RES
P=/root/r2g/pie
case $shape in
  c*)  n=${shape#c}; mode=(tput --num-requests $((2*n)) --concurrency $n --max-tokens 128 --prompts-file /root/prompts/p225.json) ;;
  pf*) n=${shape#pf}; mode=(tput --num-requests $n --concurrency $n --max-tokens 1 --prompts-file /root/prompts/p609.json) ;;
  raw) mode=(latency --requests 8 --max-tokens 256 --prompts-file /root/prompts/p609.json)
       # vLLM's latency mode stamps no TTFT; its streaming client at concurrency 1 does
       [ $eng = vllm ] && mode=(tput --num-requests 8 --concurrency 1 --max-tokens 256 --prompts-file /root/prompts/p609.json) ;;
esac
cd $P/scripts/bench
if [ $eng = pie ]; then
  PYTHONPATH=$P/packages/python/client/src:$P/packages/python/server/python \
  PIE_BENCH_INFERLET_DIR=$P/examples/text-completion-bench \
  timeout 1800 /root/venv-pie/bin/python pie_bench.py "${mode[@]}" --model ${MODEL:-openai/gpt-oss-20b} --warmup 2 \
    --pretokenized-prompts --report-timing --json-out $RES/$label.json "$@" > $RES/$label.log 2>&1
else
  timeout 1800 /root/venv-vllm/bin/python vllm_bench.py "${mode[@]}" --model ${MODEL:-openai/gpt-oss-20b} --warmup 2 \
    --report-timing --json-out $RES/$label.json "$@" > $RES/$label.log 2>&1
fi
echo EXIT=$? >> $RES/$label.log
