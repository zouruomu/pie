#!/bin/bash
# #1368's parent (ee9098f60) against main, alternating, at the batching peak.
export RES=/root/r2g/evidence/results/attr
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
for r in 1 2; do
  for e in p1368-ee9098f60 base-48a36f0d8; do
    cp /root/engines/$e.so $E
    for s in c256 c512 c1024; do RUST_LOG=info /root/r2g/evidence/scripts/bench.sh pie $s pie-$e-$s-r$r; done
  done
done
cp /root/engines/base-48a36f0d8.so $E
echo done > $RES/exp3.done
