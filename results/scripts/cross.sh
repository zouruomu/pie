#!/bin/bash
# Cross-model base vs fix: Qwen3.8-Flash-Next mini (selected attention, the changed kernel) and Qwen3-0.6B
# (the dense tensor-core decode's clusters, whose helpers moved). Raw workload, alternating.
S=/root/r2g/evidence/results/scripts
SO=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
Q38=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/mini-e16
use() { cp /root/engines/$1.so $SO; }
for r in "$@"; do
  for e in base fix; do
    use $e; MODEL=$Q38 PROMPTS=/root/prompts-512-q38.json $S/bench.sh pie x-q38-$e-r$r
    use $e; MODEL=Qwen/Qwen3-0.6B PROMPTS=/root/prompts-512-q06.json $S/bench.sh pie x-q06-$e-r$r
  done
done
use fix
