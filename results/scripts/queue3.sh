#!/bin/bash
# equal-token pie traces (512 tokens of pie's rendering a prompt) at pf256
P=/root/r2g/evidence/results/prof
for m in 120b 20b; do
  /root/r2g/scripts/prof-pie-pb.sh openai/gpt-oss-$m 256 $P pie-$m-pf256-t512 --prompts-file /root/prompts-gptoss-576.json
done
echo QUEUE3 DONE
