#!/bin/bash
# seat_to_end without the cap over the curve, and its gate oversubscribed (PIE_TO_END_OVER, in main's to_end.rs).
D=/root/r2g/evidence/scripts; X=/root/r2g/evidence/results/exp
while ps -eo args | grep -Eq "^[^ ]*python[^ ]* (vllm|pie)_bench.py"; do sleep 5; done
A="--dump-all-token-ids --engine-option diagnostics=arm-trace"
PIE_TO_END_OVER=10 $D/exp.sh exp-capenv s2e 0 c512 x0-s2e-over10-c512 $A
PIE_TO_END_OVER=20 $D/exp.sh exp-capenv s2e 0 c512 x0-s2e-over20-c512 $A
$D/exp.sh exp-capenv s2e 0 c128 x0-s2e-c128 $A
$D/exp.sh exp-capenv s2e 0 c64 x0-s2e-c64 $A
$D/exp.sh exp-capenv s2e 0 c1024 x0-s2e-c1024 $A
echo done > $X/q4.done
