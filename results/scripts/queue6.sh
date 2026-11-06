#!/bin/bash
# TRT-LLM best working config (max_batch_size 128, max_seq_len 2048, reuse off, defaults otherwise), both models, full curve
S=/root/r2g/evidence/results/scripts
R=/root/r2g/evidence/results
until [ -e $R/queue5.done ]; do sleep 15; done
export OUT=$R/trt-best
$S/trt.sh q38 trt-q38-r1 --points 64,128,256,512,1024 --max-batch-size 128 --dump-ids $OUT/trt-q38-r1.ids.json
$S/trt.sh q36 trt-q36-r1 --points 64,128,256,512,1024 --max-batch-size 128 --dump-ids $OUT/trt-q36-r1.ids.json
echo done > $R/queue6.done
