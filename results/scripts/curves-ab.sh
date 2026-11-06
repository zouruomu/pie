#!/bin/bash
# Base vs fix over the batching and prefill-batch curve, both cuts, one round each, engines alternating by cut.
S=/root/r2g/evidence/results/scripts
SO=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
A=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/mini60-e16
B=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/mini60-e16
for e in base final; do cp /root/engines/$e.so $SO; MODEL=$A $S/curve.sh a-$e; done
for e in base final; do cp /root/engines/$e.so $SO; MODEL=$B P256=/root/prompts-256-b.json P512=/root/prompts-512x64-b.json $S/curve.sh b-$e; done
cp /root/engines/final.so $SO
