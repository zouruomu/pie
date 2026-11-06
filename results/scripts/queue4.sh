#!/bin/bash
# baselines on the site prompts: pie main and vLLM, both models, round 1
S=/root/r2g/evidence/results/scripts
R=/root/r2g/evidence/results
until [ -e $R/queue3.done ]; do sleep 15; done
$S/sweep.sh $R/base2 q36 1 pie
SHAPES="c64 c128 c256" EXTRA=" " $S/sweep.sh $R/base2 q36 1 vllm
$S/sweep.sh $R/base2 q38 1 pie
SHAPES="c64 c128 c256" EXTRA=" " $S/sweep.sh $R/base2 q38 1 vllm
echo done > $R/queue4.done
