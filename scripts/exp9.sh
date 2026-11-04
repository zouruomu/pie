#!/bin/bash
# The shipped commit's engine (fa52aea7: 6f038ac7's line, the comment's numbers updated) against the
# rebased base at the shapes the line moves, with token ids, after the tests and vLLM's token reference.
until [ -f /root/r2g/evidence/results/vllm-tokens/done ]; do sleep 20; done
until grep -q EXIT=0 /root/logs/build-final.log 2>/dev/null; do sleep 20; done
cp /root/src/pie-final/packages/python/server/python/pie/_engine.abi3.so /root/engines/fix-fa52aea7.so
SHAPES="c4 c8 c16 raw" /root/r2g/evidence/scripts/sweep.sh /root/r2g/evidence/results/final2 2 pie:base-59935820d pie:fix-fa52aea7
cp /root/engines/base-59935820d.so /root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
