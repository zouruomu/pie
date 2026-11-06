#!/bin/bash
# The failing integration test(s), alone on the GPU, on fix (54385f76) and on base (e4bb499's binary, built 11-05 21:01).
T=/root/r2g/evidence/results/tests
until grep -q EXIT= $T/suite-integration.log && grep -q EXIT= $T/suite-ep.log 2>/dev/null; do sleep 15; done
for t in $(grep -o "^test [^ ]* \.\.\. FAILED" $T/suite-integration.log | awk '{print $2}'); do
  n=${t##*::}
  ( cd /root/r2g/pie-fix/crates/engine-cuda && CARGO_MANIFEST_DIR=$PWD env -u CLAUDE_CODE_MESSAGING_TOKEN \
    /root/r2g/pie-fix/target/debug/deps/engine_cuda-13d98812d365c373 --test-threads=1 --exact $t ) > $T/rerun-fix-$n.log 2>&1
  echo EXIT=$? >> $T/rerun-fix-$n.log
  ( cd /root/r2g/pie/crates/engine-cuda && CARGO_MANIFEST_DIR=$PWD env -u CLAUDE_CODE_MESSAGING_TOKEN \
    /root/r2g/pie/target/debug/deps/engine_cuda-13d98812d365c373 --test-threads=1 --exact $t ) > $T/rerun-base-$n.log 2>&1
  echo EXIT=$? >> $T/rerun-base-$n.log
done
echo done > $T/rerun.done
