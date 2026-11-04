#!/bin/bash
# After the rebase: base (new upstream main) and fix engines rebuilt from their worktrees, then
# 3 alternating rounds of base / fix / vLLM on the target cells and the fix's cells.
# usage: final.sh <base-tag> <fix-tag>   (engines in /root/engines/<tag>.so)
SHAPES="pf8 pf128 c8 c16 c256 c512 raw" /root/r2g/evidence/scripts/sweep.sh /root/r2g/evidence/results/final 3 pie:$1 pie:$2 vllm
