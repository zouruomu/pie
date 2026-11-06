#!/bin/bash
until grep -q 'QUEUE5 DONE' /root/r2g/logs/queue5.log 2>/dev/null; do sleep 10; done
W=/root/r2g/evidence/results/w4a8
mkdir -p $W
cd /root/src/pie-w4a8
env -u CLAUDE_CODE_MESSAGING_TOKEN W4A8_TMA=1 W4A8_WS=1 /usr/local/cuda/bin/nsys profile -t cuda -s none --cpuctxsw=none -f true -o $W/w4a8-gptoss-timing \
  /root/target-w4a8/release/deps/kernels_cuda-84de8ca1e1204af8 --ignored --exact linear::moe_select::w4a8_select_timing_gptoss --test-threads=1 > $W/w4a8-gptoss-timing.log 2>&1
echo EXIT=$? >> $W/w4a8-gptoss-timing.log
/usr/local/cuda/bin/nsys stats -q --report cuda_gpu_kern_sum --format csv -o $W/w4a8-gptoss-timing $W/w4a8-gptoss-timing.nsys-rep > /dev/null 2>&1
echo QUEUE6 DONE
