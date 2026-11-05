#!/bin/bash
# Verification of fix v2 (A100 profile: seat_to_end + uncapped_admission) against base e4bb499 and vLLM 0.32.0.
D=/root/r2g/evidence/scripts; R=/root/r2g/evidence/results; S=$D/sweep.sh
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
until [ -e $R/exp/q5.done ]; do sleep 10; done
A="--dump-all-token-ids --engine-option diagnostics=arm-trace"
# 1. the profile line reaches the runtime: fix v2 with uncapped_admission turned back off
$D/exp.sh fix-v2 capon - c512 fixv2-capon-c512 $A
$D/exp.sh fix-v2 none - c512 fixv2-c512 $A
# 2. Gemma-4-26B-A4B, the case #1368 was written for
MODEL=google/gemma-4-26B-A4B-it SHAPES="c512 c1024" $S $R/x26b 1 pie:base-e4bb499 pie:fix-v2 vllm
# 3. the target cell, 3 alternating rounds (base and vLLM round 1 ran earlier)
SHAPES="c64 c128 c256 c512 c1024" $S $R/base 3 pie:fix-v2 pie:base-e4bb499 vllm
# 4. cross-model
MODEL=openai/gpt-oss-120b PC=/root/prompts/oss225.json SHAPES="c256 c512 c1024" $S $R/xoss120 1 pie:base-e4bb499 pie:fix-v2 vllm
MODEL=Qwen/Qwen3-0.6B PC=/root/prompts/q165.json SHAPES="c512 c1024" $S $R/xq06 1 pie:base-e4bb499 pie:fix-v2 vllm
# 5. the rest of Gemma-4-31B's curve: low concurrency, prefill batch, raw
SHAPES="c4 c8 c16 c32 pf4 pf8 pf16 pf32 pf64 pf128 pf256 pf512 pf1024 raw" $S $R/curve 1 pie:base-e4bb499 pie:fix-v2 vllm
# 6. long outputs (1024 out at c128): the first wave seated before the gate has learned a footprint
for e in base-e4bb499 fix-v2; do
  cp /root/engines/$e.so $E
  RES=$R/long $D/bench.sh pie c128 pie-$e-c128-out1024-r1 --dump-all-token-ids --max-tokens 1024
done
echo done > $R/q6.done
