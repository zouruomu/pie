#!/bin/bash
# experiment engine (PIE_EXP_SLOT_SEATS: slots may be N x the sequences the kv pages seat at the declared context)
S=/root/r2g/evidence/results/scripts
R=/root/r2g/evidence/results
until [ -e $R/queue4.done ] && [ -e /root/wt/exp/packages/python/server/python/pie/_engine.abi3.so ]; do sleep 15; done
for c in c128 c512; do
  for n in 1 2 3 5 8; do
    PIE_EXP_SLOT_SEATS=$n PIE_PY=/root/wt/exp/packages/python/server/python RES=$R/exp-seats $S/bench.sh pie q36 $c exp-q36-$c-x$n
  done
done
echo done > $R/queue5.done
