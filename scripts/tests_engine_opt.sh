#!/bin/bash
# engine-cuda's GPU suite on the fix tree, single-threaded, from a test binary built with the dev
# profile at opt-level 2 (debug assertions and overflow checks stay on, as in a plain `cargo test`):
# the debug binary spent over an hour each in tests 21 and 23. Separate target dir; no repo change.
unset CLAUDE_CODE_MESSAGING_TOKEN
out=/root/r2g/evidence/results/tests
cd /root/r2g/pie
export CARGO_TARGET_DIR=/root/r2g/target-opt CARGO_PROFILE_DEV_OPT_LEVEL=2
f=$out/engine-cuda-fix-opt.log
echo "tree $(git log --oneline -1); CARGO_PROFILE_DEV_OPT_LEVEL=2 (debug assertions on)" > $f
taskset -c 0-31,64-95 cargo test -p engine-cuda --features cuda --test engine_cuda --no-run > $out/build-engine-opt.log 2>&1
echo EXIT=$? >> $out/build-engine-opt.log
taskset -c 0-31,64-95 cargo test -p engine-cuda --features cuda --test engine_cuda -- --test-threads=1 >> $f 2>&1
echo EXIT=$? >> $f
echo done > $out/engine-opt.done
