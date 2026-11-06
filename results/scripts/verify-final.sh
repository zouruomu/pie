#!/bin/bash
# Final-diff verification (engine `final`): c1 A/B rounds, the curves, cross-model. Then the GPU suite.
S=/root/r2g/evidence/results/scripts
SO=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
A=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/mini60-e16
B=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/mini60-e16
Q38=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/mini-e16
use() { cp /root/engines/$1.so $SO; }
for r in 1 2 3; do
  use base;  MODEL=$A PROMPTS=/root/prompts-512.json $S/bench.sh pie fin-a-base-r$r
  use final; MODEL=$A PROMPTS=/root/prompts-512.json $S/bench.sh pie fin-a-final-r$r
  use base;  MODEL=$B PROMPTS=/root/prompts-512-m3.json $S/bench.sh pie fin-b-base-r$r
  use final; MODEL=$B PROMPTS=/root/prompts-512-m3.json $S/bench.sh pie fin-b-final-r$r
done
$S/curves-ab.sh
for r in 1 2; do
  use final; MODEL=$Q38 PROMPTS=/root/prompts-512-q38.json $S/bench.sh pie x-q38-final-r$r
  use final; MODEL=Qwen/Qwen3-0.6B PROMPTS=/root/prompts-512-q06.json $S/bench.sh pie x-q06-final-r$r
done
use final
$S/gpu-suite.sh
