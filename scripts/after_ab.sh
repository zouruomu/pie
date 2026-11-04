#!/bin/bash
# When the A/B is done: engine-cuda's GPU suite on the fix tree, and kernels-cuda's moe_select on
# both trees beside it (each single-threaded).
until [ -e /root/r2g/evidence/results/ab/sweep.done ]; do sleep 15; done
/root/r2g/evidence/scripts/tests_engine.sh &
/root/r2g/evidence/scripts/tests_moe.sh &
wait
