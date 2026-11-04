#!/bin/bash
# Fix validation: base vs fix engines alternating, full greedy token ids dumped.
until [ -f /root/r2g/evidence/results/attr/exp3.done ]; do sleep 10; done
until grep -q EXIT=0 /root/logs/build-fix.log 2>/dev/null; do sleep 10; done
cp /root/src/pie-fix/packages/python/server/python/pie/_engine.abi3.so /root/engines/fix-003d8ba06.so
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
B=/root/r2g/evidence/scripts/bench.sh
export RES=/root/r2g/evidence/results/fixab
for r in 1 2; do
  for e in base-48a36f0d8 fix-003d8ba06; do
    cp /root/engines/$e.so $E; t=${e%%-*}
    for s in c4 c8 c16 c32 c64 c128 pf8 raw; do $B pie $s oss20-$t-$s-r$r --dump-all-token-ids; done
    for s in c8 c16 c32 c64 c128 pf8; do MODEL=/root/models/oss120-mini $B pie $s oss120mini-$t-$s-r$r --dump-all-token-ids; done
    for s in c16 c64 c128 pf8 raw; do MODEL=/root/models/mimo-mini $B pie $s mimomini-$t-$s-r$r --dump-all-token-ids; done
  done
done
cp /root/engines/base-48a36f0d8.so $E
echo done > $RES/exp4.done
