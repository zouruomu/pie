#!/bin/bash
until grep -q 'QUEUE7 DONE' /root/r2g/logs/queue7.log 2>/dev/null; do sleep 10; done
B=/root/r2g/evidence/results/baseline
for r in 2 3 4; do for n in 8 16; do /root/r2g/scripts/pie-pb.sh openai/gpt-oss-20b $n $B pie-20b-pf$n-r$r; done; done
echo QUEUE8 DONE
