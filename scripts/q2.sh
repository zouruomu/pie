#!/bin/bash
# Cap/seat_to_end experiments at c512, after pie's round 1; q1's later rounds wait.
B=/root/r2g/evidence/results/base; D=/root/r2g/evidence/scripts
X="exp-capenv"
until [ -e $B/pie-base-e4bb499-c1024-r1.json ]; do sleep 10; done
$D/exp.sh $X s2e - c512 x5-s2e-c512 --dump-all-token-ids --engine-option diagnostics=arm-trace
$D/exp.sh $X none 0 c512 x0-c512 --dump-all-token-ids --engine-option diagnostics=arm-trace
$D/exp.sh $X s2e 0 c512 x0-s2e-c512 --dump-all-token-ids --engine-option diagnostics=arm-trace
$D/exp.sh $X none 7 c512 x7-c512 --dump-all-token-ids --engine-option diagnostics=arm-trace
$D/exp.sh $X s2e 7 c512 x7-s2e-c512 --dump-all-token-ids --engine-option diagnostics=arm-trace
$D/exp.sh $X none - c512 x5-c512 --dump-all-token-ids --engine-option diagnostics=arm-trace
$D/exp.sh $X none - c64 x5-c64 --dump-all-token-ids --engine-option diagnostics=arm-trace
echo done > /root/r2g/evidence/results/exp/q2.done
rm -f /root/r2g/hold
