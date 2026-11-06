#!/bin/bash
# Review 1 item 1: two more traced 54385f76 Gemma-4-26B c512 runs, to catch a stalled one with pie::kv debug on.
D=/root/r2g/evidence/scripts; R=/root/r2g/evidence/results/x26b
cp /root/engines/fix-54385f7.so /root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so; rm -f /root/.pie/profiles/cuda/user/nvidia-a100-sxm4-80gb.toml
for r in 2 3; do
  PIE_CONTENTION_TRACE_EVENTS=1 RUST_LOG=warn,runtime::bootstrap=info,pie::kv=debug,runtime::planner=debug \
    MODEL=google/gemma-4-26B-A4B-it RES=$R $D/bench.sh pie c512 rv1trace-fix-54385f7-c512-r$r --dump-all-token-ids
done
echo done > $R/q17.done
