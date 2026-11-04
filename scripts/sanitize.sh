#!/bin/bash
# note-1: after both moe_select runs, the_routed_selects_answer_the_routed_dot alone on the fix tree,
# CUDA_LAUNCH_BLOCKING=1 under compute-sanitizer (memcheck), capped at 20 minutes. If the full
# memcheck runs out, again with only the pipe kernel instrumented.
unset CLAUDE_CODE_MESSAGING_TOKEN
until [ -e /root/r2g/evidence/results/tests/moe.done ]; do sleep 15; done
out=/root/r2g/evidence/results/tests
cd /root/r2g/pie
bin=$(ls $PWD/target/debug/deps/kernels_cuda-* | grep -v '\.d$' | head -1)
head="tree $(git log --oneline -1)"
cd crates/kernels-cuda  # where cargo runs the binary
t=linear::moe_select::the_routed_selects_answer_the_routed_dot
f=$out/sanitizer-fix.log
echo "$head; binary $bin; $(compute-sanitizer --version | tail -1)" > $f
CUDA_LAUNCH_BLOCKING=1 timeout 1200 taskset -c 32-63,96-127 compute-sanitizer --tool memcheck \
  $bin $t --exact --test-threads=1 --nocapture >> $f 2>&1
echo EXIT=$? >> $f
if grep -q '^EXIT=124' $f; then
  f=$out/sanitizer-fix-pipe.log
  echo "$head; only kernels matching mxfp4_mma_pipe instrumented" > $f
  CUDA_LAUNCH_BLOCKING=1 timeout 1200 taskset -c 32-63,96-127 compute-sanitizer --tool memcheck \
    --kernel-name regex:mxfp4_mma_pipe $bin $t --exact --test-threads=1 --nocapture >> $f 2>&1
  echo EXIT=$? >> $f
fi
echo done > $out/sanitizer.done
