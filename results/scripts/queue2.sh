#!/bin/bash
# site commit ee9098f60 vs main on Qwen3.6, alternating, c128 and c256
S=/root/r2g/evidence/results/scripts
R=/root/r2g/evidence/results
until [ -e $R/queue1.done ]; do sleep 15; done
for r in 1 2; do
  for c in c128 c256; do
    PIE_PY=/root/wt/ee9098f60/packages/python/server/python RES=$R/site-commit $S/bench.sh pie q36 $c ee9098-q36-$c-r$r
    RES=$R/site-commit $S/bench.sh pie q36 $c main-q36-$c-r$r
  done
done
echo done > $R/queue2.done
