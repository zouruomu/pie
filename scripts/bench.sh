#!/bin/bash
# usage: bench.sh <engine pie|vllm|sglang> <out-label> <model> <mode: tput|latency|pf> <n> [extra args...]
#   tput  n = concurrency (128 in / 128 out, max(256, 2n) requests)
#   pf    n = batch (512 in / 1 out, n requests all at once... closed loop at concurrency n, 2n requests)
#   raw   latency: 512 in / 256 out, 8 requests
eng=$1; label=$2; model=$3; mode=$4; n=$5; shift 5
R=/root/r2g/evidence/results; mkdir -p $R
B=/root/r2g/pie/scripts/bench
case $model in
  120b) M=/root/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a;;
  20b) M=/root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee;;
  *) M=$model;;
esac
case $mode in
  tput) num=$(( n*2 > 256 ? n*2 : 256 )); args=(tput --concurrency $n --num-requests $num --max-tokens 128 --prompts-file /root/prompts/p128.json --warmup 2);;
  pf) num=$(( n > 256 ? n : 256 )); args=(tput --concurrency $n --num-requests $num --max-tokens 1 --prompts-file /root/prompts/p512.json --warmup 2);;
  raw) args=(latency --requests 8 --warmup 2 --max-tokens 256 --prompts-file /root/prompts/p512.json --report-timing);;
esac
cd $B
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader > $R/$label.clk
if [ $eng = pie ]; then
  env -u CLAUDE_CODE_MESSAGING_TOKEN PYTHONPATH=/root/r2g/pie/packages/python/client/src:/root/r2g/pie/packages/python/server/python \
  PIE_BENCH_INFERLET_DIR=/root/r2g/pie/examples/text-completion-bench \
  /root/venv-pie/bin/python pie_bench.py "${args[@]}" --model $M --json-out $R/$label.json "$@" > $R/$label.log 2>&1
else
  env -u CLAUDE_CODE_MESSAGING_TOKEN /root/venv-$eng/bin/python ${eng}_bench.py "${args[@]}" --model $M --json-out $R/$label.json "$@" > $R/$label.log 2>&1
fi
echo EXIT=$? >> $R/$label.log
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader >> $R/$label.clk
