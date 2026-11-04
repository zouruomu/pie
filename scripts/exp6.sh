#!/bin/bash
# Verification sweep: round 1 the fix alone (base and vLLM have results/base r1), rounds 2-3 base, fix, vLLM.
until [ -f /root/r2g/evidence/results/xmodel/exp4.done ]; do sleep 10; done
ONLY=pie-fix-003d8ba06 /root/r2g/evidence/scripts/sweep.sh /root/r2g/evidence/results/verify 1 pie:fix-003d8ba06
mv /root/r2g/evidence/results/verify/sweep.done /root/r2g/evidence/results/verify/round1.done
/root/r2g/evidence/scripts/sweep.sh /root/r2g/evidence/results/verify2 2 pie:base-48a36f0d8 pie:fix-003d8ba06 vllm
cp /root/engines/base-48a36f0d8.so /root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
