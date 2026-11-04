#!/bin/bash
# Admission/pool knobs at the batching peak (main engine), with the pool and cap logged.
until [ -f /root/r2g/evidence/results/attr/exp3.done ]; do sleep 10; done
export RES=/root/r2g/evidence/results/knobs2
for k in rec512 nosf s2e ws55; do
  for s in c256 c512; do RUST_LOG=info /root/r2g/evidence/scripts/knob.sh $k pie $s pie-$k-$s-r1; done
done
echo done > $RES/exp5.done
