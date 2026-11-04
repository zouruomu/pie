#!/bin/bash
# kernels-cuda's routed-select tests (the MXFP4 kernels the line chooses between) on both rebased
# trees, single-threaded, pinned to the CPUs the bench does not use.
out=/root/r2g/evidence/results/tests
for t in base fix; do
  cd /root/src/pie-$t
  f=$out/kernels-moe-select-$t.log
  echo "tree $(git log --oneline -1)" > $f
  taskset -c 32-63,96-127 cargo test -p kernels-cuda --features cuda --test kernels_cuda -- linear::moe_select --test-threads=1 >> $f 2>&1
  echo EXIT=$? >> $f
done
