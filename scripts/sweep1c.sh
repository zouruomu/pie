#!/bin/bash
S=/root/r2g/evidence/scripts/bench.sh
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
until [ -f /root/r2g/evidence/results/sweep1b.done ]; do sleep 20; done
cp /root/engines/win1.so $E
for c in 32 128 256 512 1024; do $S pie r1-win1-c$c 120b tput $c --pretokenized-prompts; done
cp /root/engines/base-e4bb499c.so $E
echo done > /root/r2g/evidence/results/sweep1c.done
