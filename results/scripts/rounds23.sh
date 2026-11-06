#!/bin/bash
# rounds 2-3 at the large pf points, engines alternating, one at a time.
source /root/r2g/scripts/paths.sh
R=/root/r2g/evidence/results/baseline
for r in 2 3; do
  for m in 20b 120b; do
    for n in 128 256 512 1024; do /root/r2g/scripts/pie-pb.sh openai/gpt-oss-$m $n $R pie-$m-pf$n-r$r; done
    M=$M20; [ $m = 120b ] && M=$M120
    (cd /tmp && env -u CLAUDE_CODE_MESSAGING_TOKEN /root/venv-trtllm/bin/python -I /root/r2g/scripts/trtllm_pbatch.py --model $M \
      --points 128,256,512,1024 --prompts-file /root/prompts-gptoss-512.json --json-out $R/trt-$m-big-r$r.json > $R/trt-$m-big-r$r.log 2>&1)
  done
done
echo ROUNDS23 DONE
