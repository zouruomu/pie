#!/bin/bash
until grep -q 'ROUNDS23 DONE' /root/r2g/logs/rounds23.log; do sleep 10; done
/root/r2g/scripts/queue2.sh > /root/r2g/logs/queue2.log 2>&1
/root/r2g/scripts/sweep_vllm.sh > /root/r2g/logs/vllm-sweep.log 2>&1
