#!/bin/bash
until grep -q 'QUEUE4 DONE' /root/r2g/logs/queue4.log 2>/dev/null; do sleep 10; done
source /root/r2g/scripts/paths.sh
B=/root/r2g/evidence/results/batching
mkdir -p $B
cd /root/r2g/pie/scripts/bench
for n in 64 256 1024; do
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader > $B/pie-120b-c$n.clocks
  env -u CLAUDE_CODE_MESSAGING_TOKEN PYTHONPATH=/root/r2g/pie/packages/python/server/python:/root/r2g/pie/packages/python/client/src \
  PIE_BENCH_INFERLET_DIR=/root/r2g/pie/examples/text-completion-bench \
  /root/venv-pie/bin/python pie_bench.py tput --model openai/gpt-oss-120b --num-requests $n --concurrency $n --warmup 16 \
    --max-tokens 128 --prompts-file /root/prompts-gptoss-192.json --json-out $B/pie-120b-c$n.json > $B/pie-120b-c$n.log 2>&1
done
(cd /tmp && env -u CLAUDE_CODE_MESSAGING_TOKEN /root/venv-trtllm/bin/python -I /root/r2g/scripts/trtllm_pbatch.py --model $M120 \
  --points 64,256,1024 --max-tokens 128 --prompts-file /root/prompts-gptoss-128.json --json-out $B/trt-120b-batching.json > $B/trt-120b-batching.log 2>&1)
echo QUEUE5 DONE
