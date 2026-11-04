#!/bin/bash
# engine-cuda's GPU integration suite (`--test engine_cuda`), single-threaded, on the fix tree.
unset CLAUDE_CODE_MESSAGING_TOKEN
out=/root/r2g/evidence/results/tests
cd /root/r2g/pie
f=$out/engine-cuda-fix.log
echo "tree $(git log --oneline -1)" > $f
taskset -c 32-63,96-127 cargo test -p engine-cuda --features cuda --test engine_cuda -- --test-threads=1 >> $f 2>&1
echo EXIT=$? >> $f
echo done > $out/engine.done
