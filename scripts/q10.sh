#!/bin/bash
# The pool fit's inputs printed (PIE_EXP_TRACE, exp engine with the base's admission) at c512, 16 requests, while q8 is held.
D=/root/r2g/evidence/scripts
printf '%s\n' '^(pie|vllm)-' > /root/r2g/hold
while ps -eo args | grep -Eq "^[^ ]*python[^ ]* (vllm|pie)_bench.py"; do sleep 5; done
PIE_EXP_TRACE=1 $D/exp.sh exp-fit none - c512 fit-trace-c512 --num-requests 16 --engine-option diagnostics=arm-trace
PIE_EXP_TRACE=1 $D/exp.sh exp-fit ws40 - c512 fit-trace-ws40-c512 --num-requests 16 --engine-option diagnostics=arm-trace
rm -f /root/r2g/hold
