#!/bin/bash
# engine-cuda tests on 54385f76, the three binaries cargo built at 09:29 (CARGO_PROFILE_DEV_OPT_LEVEL=2), run
# directly: the lib unit tests (CPU-bound) beside the GPU integration binaries, each --test-threads=1.
T=/root/r2g/evidence/results/tests; B=/root/r2g/pie-fix/target/debug/deps
cd /root/r2g/pie-fix/crates/engine-cuda
export CARGO_MANIFEST_DIR=/root/r2g/pie-fix/crates/engine-cuda
run() { { echo "tree 54385f76 binary $1"; env -u CLAUDE_CODE_MESSAGING_TOKEN $B/$1 --test-threads=1; echo EXIT=$?; } > $T/$2 2>&1; }
run engine_cuda-0c6fd18c94ae260b suite-lib.log &
run engine_cuda-13d98812d365c373 suite-integration.log
run engine_cuda_ep-073d082661b5e0e2 suite-ep.log
wait
echo done > $T/suite.done
