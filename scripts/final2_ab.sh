#!/bin/bash
# Rounds 2-3 of the final pass (round 1 in results/final showed c24 -4.0% and c32 -6.3% from single
# runs; c32's decode selects keep their kernel): the rebased base and fix engines at c8, c16, c24 and
# c32, alternating, token ids dumped, after the idle-GPU test reruns. Files are named r1-r2 here.
until [ -e /root/r2g/evidence/results/tests/idle.done ]; do sleep 15; done
export SHAPES="c8 c16 c24 c32"
taskset -c 0-31,64-95 /root/r2g/evidence/scripts/sweep.sh /root/r2g/evidence/results/final2 2 pie:base-48399500 pie:fix-8e8b19b0
