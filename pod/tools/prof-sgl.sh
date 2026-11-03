#!/bin/bash
# usage: prof-sgl.sh <label> <graph-trace: node|graph>
source /etc/pod/env.sh
label=$1; gt=$2; shift 2
mkdir -p /root/results/analysis
cd /root
/usr/local/cuda/bin/nsys profile -t cuda,nvtx,osrt -s none --cpuctxsw=none --cuda-graph-trace=$gt \
  --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o /root/results/analysis/$label \
/root/venv-sglang/bin/python /root/sgl_prof.py "$@" > /root/results/analysis/$label.log 2>&1
echo EXIT=$? >> /root/results/analysis/$label.log
