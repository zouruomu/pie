#!/bin/bash
# Final pass after the rebase onto 48399500 (#1429): 1 round of the rebased base and fix engines at
# c8, c16, c24 and c32, token ids dumped, once the GPU is idle (the engine-cuda suite and the joined
# test's rerun done).
t=/root/r2g/evidence/results/tests
until grep -q '^EXIT=' $t/engine-cuda-fix-opt.log && [ -e $t/joined2.done ]; do sleep 15; done
export SHAPES="c8 c16 c24 c32"
taskset -c 0-31,64-95 /root/r2g/evidence/scripts/sweep.sh /root/r2g/evidence/results/final 1 pie:base-48399500 pie:fix-8e8b19b0
