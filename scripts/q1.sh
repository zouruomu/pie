#!/bin/bash
# Early-verdict queue: after vLLM round 1, a pie diagnostic boot (arm-trace) at c512, then
# pie round 1 and alternating rounds 2-3 of pie and vLLM over c64-c1024.
S=/root/r2g/evidence/scripts; B=/root/r2g/evidence/results/base
while pgrep -f "sweep.sh $B 1 vllm" > /dev/null; do sleep 10; done
cp /root/engines/base-e4bb499.so /root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
RES=/root/r2g/evidence/results/diag $S/bench.sh pie c512 pie-base-e4bb499-c512-armtrace --engine-option diagnostics=arm-trace
$S/sweep.sh $B 1 pie:base-e4bb499
$S/sweep.sh $B 3 vllm pie:base-e4bb499
echo done > $B/q1.done
