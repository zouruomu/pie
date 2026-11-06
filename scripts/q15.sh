#!/bin/bash
# Review 1 item 1: Gemma-4-26B-A4B c512, main (a6547509) and 54385f76 alternating, 2 rounds, for the ITL p99 spread.
D=/root/r2g/evidence/scripts
for r in 1 2; do
  for e in base-a654750 fix-54385f7; do
    MODEL=google/gemma-4-26B-A4B-it RES=/root/r2g/evidence/results/x26b $D/exp.sh $e none - c512 rv1-$e-c512-r$r --dump-all-token-ids
  done
done
echo done > /root/r2g/evidence/results/x26b/q15.done
