#!/bin/bash
# Early-verdict queue: a pie diagnostic boot (arm-trace) at c512, pie round 1, then
# alternating rounds 2-3 of vLLM and pie over c64-c1024 (vLLM round 1 ran first).
S=/root/r2g/evidence/scripts; B=/root/r2g/evidence/results/base
cp /root/engines/base-e4bb499.so /root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
RES=/root/r2g/evidence/results/diag $S/bench.sh pie c512 pie-base-e4bb499-c512-armtrace --engine-option diagnostics=arm-trace
$S/sweep.sh $B 3 vllm pie:base-e4bb499
echo done > $B/q1.done
