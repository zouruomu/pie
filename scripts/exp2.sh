#!/bin/bash
# The measured commit's engine (8fb7e90e6) on the site's shapes, then back to main's.
until [ -f /root/r2g/evidence/results/knobs/exp1.done ]; do sleep 10; done
export RES=/root/r2g/evidence/results/old
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
cp /root/engines/old-8fb7e90e6.so $E
for s in pf8 pf64 pf256 c16 c128 c256 c512 c1024; do
  RUST_LOG=info /root/r2g/evidence/scripts/bench.sh pie $s pie-old-8fb7e90e6-$s-r1
done
cp /root/engines/base-48a36f0d8.so $E
echo done > $RES/exp2.done
