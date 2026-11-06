#!/bin/bash
# Alternating raw rounds on the MiniMax-M3-MXFP8 mini60-e16 cut: pie, vLLM 0.32.0, SGLang 0.5.22.
export MODEL=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/mini60-e16 PROMPTS=/root/prompts-512-m3.json
S=/root/r2g/evidence/results/scripts
for r in "$@"; do
  $S/bench.sh pie pie-m3-r$r
  $S/bench.sh vllm vllm-m3-r${r}s
  $S/bench.sh sglang sglang-m3-r$r
done
