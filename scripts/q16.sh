#!/bin/bash
# Review 1 item 1: one traced Gemma-4-26B c512 run per engine (54385f76, then main a6547509): every pause (suspension) and residency copy logged,
# to align with the JSON's per-token arrival times.
D=/root/r2g/evidence/scripts; R=/root/r2g/evidence/results/x26b
until [ -e $R/q15.done ]; do sleep 10; done
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
for e in fix-54385f7 base-a654750; do
  cp /root/engines/$e.so $E; rm -f /root/.pie/profiles/cuda/user/nvidia-a100-sxm4-80gb.toml
  PIE_CONTENTION_TRACE_EVENTS=1 RUST_LOG=warn,runtime::bootstrap=info,pie::kv=debug,runtime::planner=debug \
    MODEL=google/gemma-4-26B-A4B-it RES=$R $D/bench.sh pie c512 rv1trace-$e-c512 --dump-all-token-ids
done
echo done > $R/q16.done
