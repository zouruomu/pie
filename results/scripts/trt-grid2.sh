#!/bin/bash
# TRT-LLM configs that start on an 80 GB card with Qwen3.6-27B bf16 (defaults do not: see report)
S=/root/r2g/evidence/results/scripts
export OUT=/root/r2g/evidence/results/trt/grid
P=${P:-128,256,512}
$S/trt.sh q36 grid-mb128-f70 --points $P --max-batch-size 128 --mem-frac 0.70
$S/trt.sh q36 grid-mb256-f85-t4096 --points $P --max-batch-size 256 --mem-frac 0.85 --max-num-tokens 4096
$S/trt.sh q36 grid-mb192-f80-t4096 --points $P --max-batch-size 192 --mem-frac 0.80 --max-num-tokens 4096
echo done > /root/r2g/evidence/results/trt/grid1.done
