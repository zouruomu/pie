#!/bin/bash
# Queued after the verification sweep: the post-rebase A/B, then the test suites on both rebased trees.
until [ -f /root/r2g/evidence/results/verify2/sweep.done ]; do sleep 30; done
/root/r2g/evidence/scripts/final.sh base-59935820d fix-6f038ac7
cp /root/engines/base-59935820d.so /root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
/root/r2g/evidence/scripts/tests.sh
