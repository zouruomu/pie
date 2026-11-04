#!/bin/bash
# The fix tree's suites beside the base tree's (each single-threaded; together under 2 GiB of the
# card), then base's routed-select test once its engine-cuda run (pid $1) is done. A failure seen here
# is rerun alone before it counts.
out=/root/r2g/evidence/results/tests
cd /root/src/pie-fix
echo "tree $(git log --oneline -1); beside the base tree's run" > $out/engine-cuda-fix.log
cargo test -p engine-cuda --features cuda --test engine_cuda -- --test-threads=1 >> $out/engine-cuda-fix.log 2>&1
echo EXIT=$? >> $out/engine-cuda-fix.log
while kill -0 $1 2>/dev/null; do sleep 10; done
echo EXIT=see-cargo >> $out/engine-cuda-base.log
for t in base fix; do
  cd /root/src/pie-$t
  echo "tree $(git log --oneline -1)" > $out/kernels-moe-$t.log
  cargo test -p kernels-cuda --features cuda --test kernels_cuda -- linear::moe_select --test-threads=1 >> $out/kernels-moe-$t.log 2>&1
  echo EXIT=$? >> $out/kernels-moe-$t.log
done
echo done > $out/tests.done
