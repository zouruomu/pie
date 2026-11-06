#!/bin/bash
# vLLM's greedy token ids on the c512 batching prompts (its offline path: the streaming client stores no ids),
# as an outside reference for pie base against fix.
R=/root/r2g/evidence/results
until [ -e $R/q8.done ]; do sleep 20; done
mkdir -p $R/vllm-tokens; cd /root/r2g/pie/scripts/bench
env -u CLAUDE_CODE_MESSAGING_TOKEN /root/venv-vllm/bin/python vllm_bench.py tput --num-requests 1024 --concurrency 512 \
  --max-tokens 128 --prompts-file /root/prompts/p169.json --model google/gemma-4-31B-it --warmup 2 --dump-all-token-ids \
  --json-out $R/vllm-tokens/vllm-offline-c512.json > $R/vllm-tokens/vllm-offline-c512.log 2>&1
echo EXIT=$? >> $R/vllm-tokens/vllm-offline-c512.log
echo done > $R/q12.done
