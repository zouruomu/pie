#!/bin/bash
until [ -e /root/r2g/evidence/results/trt/grid1.done ]; do sleep 15; done
EXTRA=" " /root/r2g/evidence/results/scripts/sweep.sh /root/r2g/evidence/results/base q36 1 vllm
