#!/bin/bash
# usage: exp.sh <engine name in /root/engines> <knobfile|none> <cap multiplier|-> <shape> <label> [bench args...]
# One run with an engine, an A100 user profile layer (removed after), and PIE_EXP_CAP_MULT (exp engine only).
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
u=/root/.pie/profiles/cuda/user/nvidia-a100-sxm4-80gb.toml
so=$1; k=$2; m=$3; shape=$4; label=$5; shift 5
cp /root/engines/$so.so $E
rm -f $u
if [ "$k" != none ]; then mkdir -p $(dirname $u); cp /root/r2g/evidence/scripts/knobs/$k.toml $u; fi
if [ "$m" != - ]; then export PIE_EXP_CAP_MULT=$m; fi
export RUST_LOG=warn,runtime::bootstrap=info
RES=${RES:-/root/r2g/evidence/results/exp} /root/r2g/evidence/scripts/bench.sh pie $shape $label "$@"
rm -f $u
