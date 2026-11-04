#!/bin/bash
# The routed test alone on the fix tree with pie's fire trace (as run for results/tests/firetrace-fix.log
# on 2f4f9cf0's binary).
unset CLAUDE_CODE_MESSAGING_TOKEN
f=/root/r2g/evidence/results/tests/firetrace-fix.log
cd /root/r2g/pie/crates/kernels-cuda
bin=$(ls /root/r2g/pie/target/debug/deps/kernels_cuda-* | grep -v '\.d$' | head -1)
echo "tree $(git log --oneline -1); PIE_CUDA_TRACE_FIRES=1 CUDA_LAUNCH_BLOCKING=1, no sanitizer" > $f
PIE_CUDA_TRACE_FIRES=1 CUDA_LAUNCH_BLOCKING=1 taskset -c 32-63,96-127 timeout 900 $bin \
  linear::moe_select::the_routed_selects_answer_the_routed_dot --exact --test-threads=1 --nocapture >> $f 2>&1
