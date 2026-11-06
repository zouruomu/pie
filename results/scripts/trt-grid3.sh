#!/bin/bash
# TRT-LLM with max_seq_len 2048 (as pie's and vLLM's max_model_len); batch caps between 128 and 256
# (256 leaves the state slots nearly the whole cache quota: 89 tok/s at c128, TTFT 179 s)
S=/root/r2g/evidence/results/scripts
export OUT=/root/r2g/evidence/results/trt/grid
P=${P:-128,256,512}
$S/trt.sh q36 grid-msl2048-mb160 --points $P --max-batch-size 160
$S/trt.sh q36 grid-msl2048-mb192 --points $P --max-batch-size 192
$S/trt.sh q36 grid-msl2048-mb224 --points $P --max-batch-size 224
$S/trt.sh q36 grid-msl2048-mb128 --points $P --max-batch-size 128
echo done > /root/r2g/evidence/results/trt/grid1.done
