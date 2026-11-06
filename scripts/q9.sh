#!/bin/bash
# Gemma-4-26B-A4B c512 repeats for base, v2, v3 while q8 is held: does working_set_percent 40 cost it?
D=/root/r2g/evidence/scripts
while ps -eo args | grep -Eq "^[^ ]*python[^ ]* (vllm|pie)_bench.py"; do sleep 5; done
for r in 2 3; do
  for e in base-e4bb499 fix-v2 fix-v3; do
    MODEL=google/gemma-4-26B-A4B-it RES=/root/r2g/evidence/results/x26b $D/exp.sh $e none - c512 xrep-$e-c512-r$r --dump-all-token-ids
  done
done
rm -f /root/r2g/hold
