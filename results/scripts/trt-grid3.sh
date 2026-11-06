#!/bin/bash
# TRT-LLM with max_seq_len 2048 (as pie's and vLLM's max_model_len): defaults otherwise, then batch caps
S=/root/r2g/evidence/results/scripts
export OUT=/root/r2g/evidence/results/trt/grid
P=${P:-128,256,512}
$S/trt.sh q36 grid-msl2048-mb256 --points $P --max-batch-size 256
$S/trt.sh q36 grid-msl2048-default --points $P
$S/trt.sh q36 grid-msl2048-mb384 --points $P,1024 --max-batch-size 384
echo done > /root/r2g/evidence/results/trt/grid1.done
