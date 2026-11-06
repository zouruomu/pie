#!/bin/bash
R=/root/r2g/evidence/results
until [ -e $R/queue2.done ]; do sleep 15; done
/root/r2g/evidence/results/scripts/trt-grid3.sh
echo done > $R/queue3.done
