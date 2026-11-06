#!/bin/bash
# The suite's failures rerun on base e2e9d1e (worktree /root/wt-base), then the two test binaries cargo
# skipped after them on the final commit.
T=/root/r2g/evidence/results/tests
export CARGO_PROFILE_DEV_OPT_LEVEL=2
cd /root/wt-base
( time env -u CLAUDE_CODE_MESSAGING_TOKEN CARGO_TARGET_DIR=/root/wt-base/target cargo test -p engine-cuda --features cuda --test engine_cuda -- a_joined_projection_answers_what_its_legs_answer --test-threads=1 ) > $T/base-a_joined_projection.log 2>&1; echo "EXIT=$?" >> $T/base-a_joined_projection.log
( time env -u CLAUDE_CODE_MESSAGING_TOKEN CARGO_TARGET_DIR=/root/wt-base/target cargo test -p kernels-cuda --features cuda --lib -- jit::nvrtc::tests::every_unit_lowers --test-threads=1 ) > $T/base-every_unit_lowers.log 2>&1; echo "EXIT=$?" >> $T/base-every_unit_lowers.log
cd /root/r2g/pie
git log --oneline -1 > $T/gpu-suite-rest-commit.txt
( time env -u CLAUDE_CODE_MESSAGING_TOKEN cargo test -p kernels-cuda --features cuda --test kernels_cuda -- --test-threads=1 ) > $T/gpu-suite-kernels-cuda-integration.log 2>&1; echo "EXIT=$?" >> $T/gpu-suite-kernels-cuda-integration.log
( time env -u CLAUDE_CODE_MESSAGING_TOKEN cargo test -p engine-cuda --features cuda --test engine_cuda_ep -- --test-threads=1 ) > $T/gpu-suite-engine-cuda-ep.log 2>&1; echo "EXIT=$?" >> $T/gpu-suite-engine-cuda-ep.log
