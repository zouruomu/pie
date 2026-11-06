#!/bin/bash
# Verification of fix v3 (A100 profile: seat_to_end + uncapped_admission + working_set_percent 40)
# against base e4bb499 and vLLM 0.32.0; fix v2 (without the working-set line) beside it where that line
# could cost (Gemma-4-26B, prefill batches).
D=/root/r2g/evidence/scripts; R=/root/r2g/evidence/results; S=$D/sweep.sh
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
while ps -eo args | grep -Eq "^[^ ]*python[^ ]* (vllm|pie)_bench.py"; do sleep 5; done
until [ -e /root/engines/fix-v3.so ]; do sleep 10; done
A="--dump-all-token-ids --engine-option diagnostics=arm-trace"
$D/exp.sh fix-v2 none - c512 fixv2-c512 $A
$D/exp.sh fix-v3 none - c512 fixv3-c512 $A
MODEL=google/gemma-4-26B-A4B-it SHAPES="c512 c1024" $S $R/x26b 1 pie:base-e4bb499 pie:fix-v3 pie:fix-v2 vllm
SHAPES="c64 c128 c256 c512 c1024" $S $R/base 3 pie:fix-v3 pie:base-e4bb499 vllm
MODEL=openai/gpt-oss-120b PC=/root/prompts/oss225.json SHAPES="c256 c512 c1024" $S $R/xoss120 1 pie:base-e4bb499 pie:fix-v3 vllm
MODEL=Qwen/Qwen3-0.6B PC=/root/prompts/q165.json SHAPES="c512 c1024" $S $R/xq06 1 pie:base-e4bb499 pie:fix-v3 vllm
SHAPES="c4 c8 c16 c32 pf4 pf8 pf16 pf32 pf64 pf128 pf256 pf512 pf1024 raw" $S $R/curve 1 pie:base-e4bb499 pie:fix-v3 vllm
SHAPES="pf32 pf64 pf128 pf256 pf512 pf1024" $S $R/curve 1 pie:fix-v2
for e in base-e4bb499 fix-v3; do
  cp /root/engines/$e.so $E
  RES=$R/long $D/bench.sh pie c128 pie-$e-c128-out1024-r1 --dump-all-token-ids --max-tokens 1024
done
echo done > $R/q7.done
