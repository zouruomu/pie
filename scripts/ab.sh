#!/bin/bash
# The final A/B on this tree: 3 alternating rounds of the base (upstream 1b851247) and fix (2f4f9cf0)
# engines at c8, c16, c24 and c32, token ids dumped. vLLM 0.32.0 is unchanged since the previous run.
export SHAPES="c8 c16 c24 c32"
/root/r2g/evidence/scripts/sweep.sh /root/r2g/evidence/results/ab 3 pie:base-1b851247 pie:fix-2f4f9cf0
