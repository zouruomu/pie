#!/bin/bash
# Final: the reviewed commit 54385f76 (on main dc6282d7) at c128/c512 beside main a6547509's runs in results/final,
# then the engine-cuda GPU suite on 54385f76, single-threaded, alone on the GPU.
D=/root/r2g/evidence/scripts; R=/root/r2g/evidence/results
until [ -e $R/final/sweep.done ]; do sleep 20; done
rm -f $R/final/sweep.done
SHAPES="c128 c512" $D/sweep.sh $R/final 1 pie:fix-54385f7
until [ -e $R/tests/final-cpu.done ]; do sleep 20; done
cd /root/r2g/pie-fix
{ echo "tree $(git rev-parse HEAD)"; env -u CLAUDE_CODE_MESSAGING_TOKEN CARGO_PROFILE_DEV_OPT_LEVEL=2 cargo test -p engine-cuda --features cuda -- --test-threads=1; echo EXIT=$?; } \
  > $R/tests/engine-cuda-gpu-fix.log 2>&1
echo done > $R/q14.done
