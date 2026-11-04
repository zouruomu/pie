#!/bin/bash
# After the rebase: 2 alternating rounds of the rebased base and fix engines on the target cells, the
# fix's cells and c24 (the 3-routes-an-expert selects); vLLM is unchanged by the rebase
# (vLLM's own 3 rounds over the whole curve are in results/base and results/verify2).
# usage: final.sh <base-tag> <fix-tag>   (engines in /root/engines/<tag>.so)
export SHAPES="pf8 c8 c16 c24 c256 c512 raw"
/root/r2g/evidence/scripts/sweep.sh /root/r2g/evidence/results/final 2 pie:$1 pie:$2


