#!/bin/bash
# The final rebase's pair (base 1b851247, fix 8fc6862c, the same diff) on the target cells, one round.
until grep -q EXIT /root/logs/build-nbase.log && grep -q EXIT /root/logs/build-nfix.log; do sleep 20; done
grep -q EXIT=0 /root/logs/build-nbase.log && grep -q EXIT=0 /root/logs/build-nfix.log || exit 1
cp /root/src/pie-nbase/packages/python/server/python/pie/_engine.abi3.so /root/engines/base-1b851247.so
cp /root/src/pie-nfix/packages/python/server/python/pie/_engine.abi3.so /root/engines/fix-8fc6862c.so
SHAPES="pf8 c8 c16 c512 raw" /root/r2g/evidence/scripts/sweep.sh /root/r2g/evidence/results/final3 1 pie:base-1b851247 pie:fix-8fc6862c
cp /root/engines/base-1b851247.so /root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
