#!/bin/bash
# The alternative lever for c8-c24: per_from kept at 4, mxfp4_grouped_reuse 0 (c8/c16 on the grouped
# wmma select instead of the by-route one), against base and the fix, on the rebased engines.
until [ -f /root/r2g/evidence/results/final/sweep.done ]; do sleep 15; done
export RES=/root/r2g/evidence/results/knobs3
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
K=/root/r2g/evidence/scripts/knob.sh
for r in 1 2; do
  cp /root/engines/base-59935820d.so $E
  for s in c4 c8 c16 c20 c24; do $K reuse0 pie $s pie-base-reuse0-$s-r$r --dump-all-token-ids; done
  for s in c20; do $K none pie $s pie-base-59935820d-$s-r$r --dump-all-token-ids; done
  cp /root/engines/fix-6f038ac7.so $E
  for s in c20; do $K none pie $s pie-fix-6f038ac7-$s-r$r --dump-all-token-ids; done
done
cp /root/engines/base-59935820d.so $E
echo done > $RES/exp8.done
/root/r2g/evidence/scripts/tests.sh
