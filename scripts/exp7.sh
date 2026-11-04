#!/bin/bash
# #1350 alone (the measured commit's engine with the 65% working set as a user layer), then the
# 32-expert MiMo miniature base vs fix, then the verification sweep.
until [ -f /root/r2g/evidence/results/xmodel/exp4.done ]; do sleep 10; done
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
B=/root/r2g/evidence/scripts/bench.sh
cp /root/engines/old-8fb7e90e6.so $E
for s in c256 c512 pf256; do RES=/root/r2g/evidence/results/old RUST_LOG=info /root/r2g/evidence/scripts/knob.sh ws65 pie $s pie-old-8fb7e90e6-ws65-$s-r1; done
export RES=/root/r2g/evidence/results/xmodel
for r in 1 2; do
  for e in base-48a36f0d8 fix-003d8ba06; do
    cp /root/engines/$e.so $E; t=${e%%-*}
    for s in c4 c8 c16 c64 pf8 raw; do MODEL=/root/models/mimo-mini-e32 $B pie $s mimoe32-$t-$s-r$r --dump-all-token-ids; done
  done
done
cp /root/engines/base-48a36f0d8.so $E
/root/r2g/evidence/scripts/exp6.sh
