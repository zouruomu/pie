#!/bin/bash
# round 1: whole batching curve per engine, then raw + pf
S=/root/r2g/evidence/scripts/bench.sh
for c in 32 64 128 256 512 1024; do $S pie r1-pie-c$c 120b tput $c --pretokenized-prompts; done
for c in 32 64 128 256 512 1024; do $S vllm r1-vllm-c$c 120b tput $c; done
for c in 32 64 128 256 512 1024; do $S sglang r1-sgl-c$c 120b tput $c --sglang-cuda-graph-max-bs $c; done
echo done > /root/r2g/evidence/results/sweep1.done
