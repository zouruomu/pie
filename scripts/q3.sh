#!/bin/bash
# After q2's x5-c512: the base-equivalent c64, then seat_to_end traced (PIE_EXP_TRACE, exp engine 2) at c512,
# then release the hold on rounds 2-3.
D=/root/r2g/evidence/scripts; X=/root/r2g/evidence/results/exp
while ps -eo args | grep -q "^/bin/bash /root/r2g/evidence/scripts/exp.sh"; do sleep 5; done
$D/exp.sh exp-capenv none - c64 x5-c64 --dump-all-token-ids --engine-option diagnostics=arm-trace
until [ -e /root/engines/exp-trace.so ]; do sleep 10; done
PIE_EXP_TRACE=1 $D/exp.sh exp-trace s2e 0 c512 x0-s2e-trace-c512 --engine-option diagnostics=arm-trace
echo done > $X/q3.done
rm -f /root/r2g/hold
