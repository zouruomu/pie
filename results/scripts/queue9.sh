#!/bin/bash
until grep -q 'QUEUE8 DONE' /root/r2g/logs/queue8.log 2>/dev/null; do sleep 10; done
B=/root/r2g/evidence/results/clocks
mkdir -p $B
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,temperature.gpu --format=csv,noheader -lms 100 > $B/pie-120b-pf1024.smi.csv &
s=$!
/root/r2g/scripts/pie-pb.sh openai/gpt-oss-120b 1024 $B pie-120b-pf1024
kill $s
echo QUEUE9 DONE
