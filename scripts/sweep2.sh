#!/bin/bash
# round 2: site-like flags (--report-timing, 4x concurrency requests), pie base vs vLLM, alternating
S=/root/r2g/evidence/scripts/bench.sh
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
until [ -f /root/r2g/evidence/results/sweep1c.done ]; do sleep 20; done
cp /root/engines/base-e4bb499c.so $E
export MULT=4
for c in 128 256 512 1024; do
  $S pie r2-pie-c$c 120b tput $c --pretokenized-prompts --report-timing
  $S vllm r2-vllm-c$c 120b tput $c --report-timing
done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $S sglang r2-sgl-c256 120b tput 256 --report-timing --sglang-cuda-graph-max-bs 256
echo done > /root/r2g/evidence/results/sweep2.done
