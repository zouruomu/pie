#!/bin/bash
# usage: knob.sh <knobfile|none> <bench.sh args...>
# Installs /root/r2g/evidence/scripts/knobs/<knobfile>.toml as the L4 user profile layer
# for one run, then removes it (a leftover layer silently changes later runs).
u=/root/.pie/profiles/cuda/user/nvidia-l4.toml
k=$1; shift
rm -f $u
if [ "$k" != none ]; then mkdir -p $(dirname $u); cp /root/r2g/evidence/scripts/knobs/$k.toml $u; fi
/root/r2g/evidence/scripts/bench.sh "$@"
rm -f $u
