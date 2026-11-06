#!/bin/bash
# one serial GPU queue (run with nohup): vLLM q36 sweep, TRT-LLM grid with max_seq_len 2048, pie context experiment
S=/root/r2g/evidence/results/scripts
R=/root/r2g/evidence/results
EXTRA=" " $S/sweep.sh $R/base q36 1 vllm
$S/trt-grid3.sh
for c in c128 c256 c512; do RES=$R/exp $S/bench.sh pie q36 $c pie-q36-$c-mml512 --max-model-len 512; done
echo done > $R/queue1.done
