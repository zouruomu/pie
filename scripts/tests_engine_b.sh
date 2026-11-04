#!/bin/bash
# engine-cuda's GPU suite, tests 22-60 of its 60 (the order libtest lists and runs them in), in a
# second single-threaded process on the same fix-tree binary. The first process
# (tests_engine.sh) was still in test 21, a_joined_projection_answers_what_its_legs_answer, after
# 55 minutes. Run from the package directory, as cargo runs it.
unset CLAUDE_CODE_MESSAGING_TOKEN
out=/root/r2g/evidence/results/tests
cd /root/r2g/pie/crates/engine-cuda
bin=/root/r2g/pie/target/debug/deps/engine_cuda-31e35b6bdf05c52f
f=$out/engine-cuda-fix-b.log
echo "tree $(git log --oneline -1); binary $bin; tests 22-60" > $f
mapfile -t names < <($bin --list 2>/dev/null | grep ': test$' | sed 's/: test$//' | sed -n '22,60p')
taskset -c 32-63,96-127 $bin --exact --test-threads=1 "${names[@]}" >> $f 2>&1
echo EXIT=$? >> $f
echo done > $out/engine-b.done
