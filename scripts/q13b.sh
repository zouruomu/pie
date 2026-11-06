#!/bin/bash
# Final pass after q12: the rebased fix (d678268d) against main a6547509 at the cell's high points, then the
# engine-cuda GPU suite on the fix tree, single-threaded, alone on the GPU.
D=/root/r2g/evidence/scripts; R=/root/r2g/evidence/results
until [ -e $R/q12.done ]; do sleep 20; done
SHAPES="c128 c512" $D/sweep.sh $R/final 1 pie:base-a654750 pie:fix-d678268
mkdir -p $R/tests
cd /root/r2g/pie-fix
env -u CLAUDE_CODE_MESSAGING_TOKEN CARGO_PROFILE_DEV_OPT_LEVEL=2 cargo test -p engine-cuda --features cuda -- --test-threads=1 \
  > $R/tests/engine-cuda-gpu-fix.log 2>&1
echo EXIT=$? >> $R/tests/engine-cuda-gpu-fix.log
echo done > $R/q13.done
