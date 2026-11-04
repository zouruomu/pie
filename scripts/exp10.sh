#!/bin/bash
# vLLM's greedy token reference (offline path, c4/c8/c16), then the shipped engine (fa52aea7) against
# the rebased base at c4/c8/c16/raw. Runs beside the CPU-bound engine-cuda test suites, which are
# pinned to CPUs 32-63,96-127 (the bench takes 0-31,64-95).
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
until grep -q EXIT=0 /root/logs/build-final.log 2>/dev/null; do sleep 20; done
cp /root/src/pie-final/packages/python/server/python/pie/_engine.abi3.so /root/engines/fix-fa52aea7.so
SHAPES="c4 c8 c16 raw" /root/r2g/evidence/scripts/sweep.sh /root/r2g/evidence/results/final2 2 pie:base-59935820d pie:fix-fa52aea7
cp /root/engines/base-59935820d.so /root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
