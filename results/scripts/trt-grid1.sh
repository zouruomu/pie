#!/bin/bash
S=/root/r2g/evidence/results/scripts
P=128,256
$S/trt.sh q36 grid-mb128-f85 --points $P --max-batch-size 128 --mem-frac 0.85
$S/trt.sh q36 grid-mb192-f85 --points $P --max-batch-size 192 --mem-frac 0.85
$S/trt.sh q36 grid-mb256-f85-t4096 --points $P --max-batch-size 256 --mem-frac 0.85 --max-num-tokens 4096
$S/trt.sh q36 grid-mb128-f85-fp32 --points $P --max-batch-size 128 --mem-frac 0.85 --ssm-dtype float32
echo done > /root/r2g/evidence/results/trt/grid1.done
