#!/bin/bash
# round 1: the whole pf sweep, pie then TRT-LLM, 20b then 120b, one engine at a time.
source /root/r2g/scripts/paths.sh
R=/root/r2g/evidence/results/baseline
mkdir -p $R
for m in 20b 120b; do
  for n in 4 8 16 32 64 128 256 512 1024; do
    /root/r2g/scripts/pie-pb.sh openai/gpt-oss-$m $n $R pie-$m-pf$n-r1
  done
  M=$M20; [ $m = 120b ] && M=$M120
  (cd /tmp && env -u CLAUDE_CODE_MESSAGING_TOKEN /root/venv-trtllm/bin/python -I /root/r2g/scripts/trtllm_pbatch.py --model $M \
    --prompts-file /root/prompts-gptoss-512.json --json-out $R/trt-$m-sweep-r1.json --dump-ids $R/trt-$m-sweep-r1.ids.json > $R/trt-$m-sweep-r1.log 2>&1)
done
echo SWEEP1 DONE
