#!/bin/bash
# Alternating A/B rounds: base vs fix engine on cut A, and base vs fix vs vLLM 0.32.0 on cut B.
# The engine .so is swapped in place before each pie run; bench.sh records its sha256 in the .meta.
S=/root/r2g/evidence/results/scripts
SO=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
A=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/mini60-e16
B=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/mini60-e16
use() { cp /root/engines/$1.so $SO; }
for r in "$@"; do
  use base; MODEL=$A PROMPTS=/root/prompts-512.json $S/bench.sh pie ab-a-base-r$r
  use fix;  MODEL=$A PROMPTS=/root/prompts-512.json $S/bench.sh pie ab-a-fix-r$r
  use base; MODEL=$B PROMPTS=/root/prompts-512-m3.json $S/bench.sh pie ab-b-base-r$r
  use fix;  MODEL=$B PROMPTS=/root/prompts-512-m3.json $S/bench.sh pie ab-b-fix-r$r
  MODEL=$B PROMPTS=/root/prompts-512-m3.json $S/bench.sh vllm ab-b-vllm-r$r
done
use fix
