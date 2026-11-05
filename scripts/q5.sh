#!/bin/bash
# Fix v1 (cap lifted under seat_to_end + A100 seat_to_end) at c512, then the working set held smaller.
D=/root/r2g/evidence/scripts; X=/root/r2g/evidence/results/exp
until [ -e $X/q4.done ]; do sleep 10; done
A="--dump-all-token-ids --engine-option diagnostics=arm-trace"
$D/exp.sh fix-v1 none - c512 fixv1-c512 $A
$D/exp.sh fix-v1 ws40 - c512 fixv1-ws40-c512 $A
$D/exp.sh fix-v1 ws30 - c512 fixv1-ws30-c512 $A
echo done > $X/q5.done
