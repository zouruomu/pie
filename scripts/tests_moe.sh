#!/bin/bash
# kernels-cuda's routed-select tests on both trees, single-threaded: the fix tree (/root/r2g/pie,
# 2f4f9cf0, with the new 32x16x4 case) and the base tree (/root/src/pie-base, upstream 1b851247).
unset CLAUDE_CODE_MESSAGING_TOKEN
out=/root/r2g/evidence/results/tests
for t in fix base; do
  d=$([ $t = fix ] && echo /root/r2g/pie || echo /root/src/pie-base)
  cd $d
  f=$out/kernels-moe-select-$t.log
  echo "tree $(git log --oneline -1)" > $f
  taskset -c 32-63,96-127 cargo test -p kernels-cuda --features cuda --test kernels_cuda -- linear::moe_select --test-threads=1 >> $f 2>&1
  echo EXIT=$? >> $f
done
echo done > $out/moe.done
