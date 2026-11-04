#!/bin/bash
# vLLM's greedy token ids on the batching prompts, as a reference for pie base vs fix. Its streaming
# client (--report-timing) stores no token ids, so this is the offline LLM.generate path, same shapes.
until [ -f /root/r2g/evidence/results/tests/tests.done ]; do sleep 30; done
source /etc/pod/env.sh
RES=/root/r2g/evidence/results/vllm-tokens; mkdir -p $RES
cd /root/r2g/pie/scripts/bench
for n in 4 8 16; do
  /root/venv-vllm/bin/python vllm_bench.py tput --num-requests $((2*n)) --concurrency $n --max-tokens 128 \
    --prompts-file /root/prompts/p225.json --model openai/gpt-oss-20b --warmup 2 --dump-all-token-ids \
    --json-out $RES/vllm-offline-c$n.json > $RES/vllm-offline-c$n.log 2>&1
  echo EXIT=$? >> $RES/vllm-offline-c$n.log
done
echo done > $RES/done
