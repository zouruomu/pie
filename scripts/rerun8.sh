#!/bin/bash
# The 8 integration tests that failed while the disk was full, again on the fix binary (54385f76), one at a time.
T=/root/r2g/evidence/results/tests
cd /root/r2g/pie-fix/crates/engine-cuda; export CARGO_MANIFEST_DIR=$PWD
{ echo "tree 54385f76 binary engine_cuda-13d98812d365c373, tests failed in the disk-full window"
  env -u CLAUDE_CODE_MESSAGING_TOKEN /root/r2g/pie-fix/target/debug/deps/engine_cuda-13d98812d365c373 --test-threads=1 --exact $(cat $T/rerun-list.txt)
  echo EXIT=$?; } > $T/rerun8-fix.log 2>&1
