#!/bin/bash
export RES=/root/r2g/evidence/results/knobs
K=/root/r2g/evidence/scripts/knob.sh
for r in 1 2; do
  for k in none per1 per0; do for s in c8 c16; do $K $k pie $s pie-$k-$s-r$r; done; done
done
for k in ws50 none; do for s in pf8 c256 c512; do $K $k pie $s pie-$k-$s-r1; done; done
echo done > $RES/exp1.done
