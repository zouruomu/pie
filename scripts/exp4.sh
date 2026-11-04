#!/bin/bash
# Cross-model check of the fix: base vs fix engines alternating, full greedy token ids dumped.
cp /root/src/pie-fix/packages/python/server/python/pie/_engine.abi3.so /root/engines/fix-003d8ba06.so
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
B=/root/r2g/evidence/scripts/bench.sh
export RES=/root/r2g/evidence/results/xmodel
for r in 1 2; do
  for e in base-48a36f0d8 fix-003d8ba06; do
    cp /root/engines/$e.so $E; t=${e%%-*}
    for s in c4 c8 c16 c32 c64 pf8 raw; do MODEL=/root/models/oss20-mini $B pie $s oss20mini-$t-$s-r$r --dump-all-token-ids; done
    for s in c16 c64 c128 pf8 raw; do MODEL=/root/models/mimo-mini $B pie $s mimomini-$t-$s-r$r --dump-all-token-ids; done
  done
done
cp /root/engines/base-48a36f0d8.so $E
echo done > $RES/exp4.done
