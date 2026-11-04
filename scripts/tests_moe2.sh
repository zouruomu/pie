#!/bin/bash
# After the case became gpt-oss's GLU (f79b846c): kernels-cuda's moe_select on the fix tree again,
# single-threaded, with pie's fire trace on, so the log shows the kernel each case fires.
unset CLAUDE_CODE_MESSAGING_TOKEN
out=/root/r2g/evidence/results/tests
cd /root/r2g/pie
f=$out/kernels-moe-select-fix2.log
echo "tree $(git log --oneline -1); PIE_CUDA_TRACE_FIRES=1" > $f
PIE_CUDA_TRACE_FIRES=1 taskset -c 32-63,96-127 cargo test -p kernels-cuda --features cuda --test kernels_cuda -- linear::moe_select --test-threads=1 --nocapture >> $f 2>&1
echo EXIT=$? >> $f
