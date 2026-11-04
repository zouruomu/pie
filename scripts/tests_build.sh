#!/bin/bash
# Compiles the test binaries (no run) on both trees while the A/B runs, pinned to the CPUs the
# bench does not use: kernels-cuda's kernels_cuda and engine-cuda's engine_cuda, debug builds.
unset CLAUDE_CODE_MESSAGING_TOKEN
out=/root/r2g/evidence/results/tests; mkdir -p $out
for t in /root/r2g/pie /root/src/pie-base; do
  cd $t
  n=$(basename $t)
  taskset -c 32-63,96-127 nice cargo test -p kernels-cuda --features cuda --test kernels_cuda --no-run > $out/build-kernels-$n.log 2>&1; echo EXIT=$? >> $out/build-kernels-$n.log
  taskset -c 32-63,96-127 nice cargo test -p engine-cuda --features cuda --test engine_cuda --no-run > $out/build-engine-$n.log 2>&1; echo EXIT=$? >> $out/build-engine-$n.log
done
echo done > $out/build.done
