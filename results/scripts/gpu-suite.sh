#!/bin/bash
# The GPU suite on the installed source tree (the fix commit): kernels-cuda then engine-cuda tests,
# single-threaded, dev profile at opt-level 2. Nothing else may use the GPU while it runs.
cd /root/r2g/pie
T=/root/r2g/evidence/results/tests
git log --oneline -1 > $T/gpu-suite-commit.txt
git status --short >> $T/gpu-suite-commit.txt
export CARGO_PROFILE_DEV_OPT_LEVEL=2
( time env -u CLAUDE_CODE_MESSAGING_TOKEN cargo test -p kernels-cuda --features cuda --tests -- --test-threads=1 ) > $T/gpu-suite-kernels-cuda.log 2>&1
echo "EXIT=$?" >> $T/gpu-suite-kernels-cuda.log
( time env -u CLAUDE_CODE_MESSAGING_TOKEN cargo test -p engine-cuda --features cuda --tests -- --test-threads=1 ) > $T/gpu-suite-engine-cuda.log 2>&1
echo "EXIT=$?" >> $T/gpu-suite-engine-cuda.log
