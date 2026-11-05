#!/bin/bash
S=/root/r2g/evidence/scripts/bench.sh
until [ -f /root/r2g/evidence/results/sweep1.done ]; do sleep 20; done
for c in 32 64 128 256 512 1024; do $S pie r1-pie-c$c 120b tput $c --pretokenized-prompts; done
echo done > /root/r2g/evidence/results/sweep1b.done
