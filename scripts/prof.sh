#!/bin/bash
# usage: prof.sh <knobfile|none> <shape> <label> <graph-trace: node|graph> <delay_s> <dur_s> [extra bench args]
# Traces stay in /root/prof-raw on the pod: nsys records the process environment (API keys) in them.
# nsys capture of a measured window of one pie run (the engine is embedded in pie_bench's process).
k=$1; shape=$2; label=$3; gt=$4; d=$5; t=$6; shift 6
export RES=/root/r2g/evidence/results/prof
mkdir -p $RES
/usr/local/cuda/bin/nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=$gt \
  --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o /root/prof-raw/$label \
  /root/r2g/evidence/scripts/knob.sh $k pie $shape $label --cuda-profiler-capture \
  --cuda-profiler-delay-s $d --cuda-profiler-duration-s $t "$@" > $RES/$label.nsys.log 2>&1
/usr/local/cuda/bin/nsys export -t sqlite -f true -o /root/prof-raw/$label.sqlite /root/prof-raw/$label.nsys-rep >> $RES/$label.nsys.log 2>&1
