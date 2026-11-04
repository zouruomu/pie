#!/bin/bash
# engine-cuda's GPU integration suite (`--test engine_cuda`; `engine_cuda_ep` wants several cards) and
# kernels-cuda's routed-select tests, single-threaded, on the rebased base (/root/src/pie-base) and fix (/root/src/pie-fix) trees, same build commands.
out=/root/r2g/evidence/results/tests; mkdir -p $out
for t in base fix; do
  cd /root/src/pie-$t
  echo "tree $(git log --oneline -1)" > $out/engine-cuda-$t.log
  cargo test -p engine-cuda --features cuda --test engine_cuda -- --test-threads=1 >> $out/engine-cuda-$t.log 2>&1
  echo EXIT=$? >> $out/engine-cuda-$t.log
  echo "tree $(git log --oneline -1)" > $out/kernels-moe-$t.log
  cargo test -p kernels-cuda --features cuda --test kernels_cuda -- linear::moe_select --test-threads=1 >> $out/kernels-moe-$t.log 2>&1
  echo EXIT=$? >> $out/kernels-moe-$t.log
done
echo done > $out/tests.done
