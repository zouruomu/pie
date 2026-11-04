#!/bin/bash
# The profile-resolution tests (engine-cuda `chip::`, kernels-cuda `tuning`) with no card visible, on
# both rebased trees, pinned to the CPUs the benchmarks do not use. The catalog-wide library tests
# (every deployment traced and fused in a debug build, hours) do not read a device profile.
out=/root/r2g/evidence/results/tests; mkdir -p $out
for t in base fix; do
  cd /root/src/pie-$t
  f=$out/profile-tests-nocard-$t.log
  echo "tree $(git log --oneline -1); CUDA_VISIBLE_DEVICES= (no card)" > $f
  CUDA_VISIBLE_DEVICES= taskset -c 32-63,96-127 cargo test -p engine-cuda --features cuda --lib chip:: -- --test-threads=1 >> $f 2>&1; echo EXIT=$? >> $f
  CUDA_VISIBLE_DEVICES= taskset -c 32-63,96-127 cargo test -p kernels-cuda --features cuda --lib tuning -- --test-threads=1 >> $f 2>&1; echo EXIT=$? >> $f
done
