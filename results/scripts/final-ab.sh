#!/bin/bash
# After rebasing the fix onto the newest upstream main: base (upstream main) vs fix (rebased), one round a cut.
S=/root/r2g/evidence/results/scripts
SO=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
A=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/mini60-e16
B=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/mini60-e16
for e in main fixr; do cp /root/engines/$e.so $SO; MODEL=$A PROMPTS=/root/prompts-512.json $S/bench.sh pie final-a-$e; done
for e in main fixr; do cp /root/engines/$e.so $SO; MODEL=$B PROMPTS=/root/prompts-512-m3.json $S/bench.sh pie final-b-$e; done
cp /root/engines/fixr.so $SO
