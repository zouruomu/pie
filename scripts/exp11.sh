#!/bin/bash
# After the shipped-engine round: vLLM's c8/c16 token references again (their first start found the
# card still held), with a pause between runs, then the routed-select tests.
until [ -f /root/r2g/evidence/results/final2/sweep.done ]; do sleep 15; done
source /etc/pod/env.sh
RES=/root/r2g/evidence/results/vllm-tokens
cd /root/r2g/pie/scripts/bench
for n in 8 16; do
  sleep 20
  /root/venv-vllm/bin/python vllm_bench.py tput --num-requests $((2*n)) --concurrency $n --max-tokens 128 \
    --prompts-file /root/prompts/p225.json --model openai/gpt-oss-20b --warmup 2 --dump-all-token-ids \
    --json-out $RES/vllm-offline-c$n.json > $RES/vllm-offline-c$n.log 2>&1
  echo EXIT=$? >> $RES/vllm-offline-c$n.log
done
/root/r2g/evidence/scripts/tests_moe.sh
echo done > /root/r2g/evidence/results/tests/moe.done
