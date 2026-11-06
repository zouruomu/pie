#!/bin/bash
# pie knob experiments after the vLLM sweep: a shorter declared context lets more state slots seat
S=/root/r2g/evidence/results/scripts
until [ -e /root/r2g/evidence/results/base/sweep-q36.done ] && ! pgrep -f q-vllm1.sh > /dev/null; do sleep 15; done
export RES=/root/r2g/evidence/results/exp
for c in c128 c256 c512; do
  $S/bench.sh pie q36 $c pie-q36-$c-mml512 --max-model-len 512
done
echo done > $RES/exp1.done
