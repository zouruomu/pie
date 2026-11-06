#!/bin/bash
source /root/r2g/scripts/paths.sh
P=/root/r2g/evidence/results/prof
B=/root/r2g/evidence/results/baseline
E=/root/r2g/evidence/results/equal-tokens
mkdir -p $P $E
# 1. TRT-LLM nsys over a pf256 point (the window is cut by the harness's NVTX range)
for m in 120b 20b; do
  M=$M20; [ $m = 120b ] && M=$M120
  (cd /tmp && env -u CLAUDE_CODE_MESSAGING_TOKEN /usr/local/cuda/bin/nsys profile -t cuda,nvtx -s none --cpuctxsw=none \
    --cuda-graph-trace=node --trace-fork-before-exec=true -f true -o $P/trt-$m-pf256 \
    /root/venv-trtllm/bin/python -I /root/r2g/scripts/trtllm_pbatch.py --model $M --points 256 \
    --prompts-file /root/prompts-gptoss-512.json --json-out $P/trt-$m-pf256.json > $P/trt-$m-pf256.log 2>&1)
  /usr/local/cuda/bin/nsys export -t sqlite -f true -o $P/trt-$m-pf256.sqlite $P/trt-$m-pf256.nsys-rep > /dev/null 2>&1
  python3 /root/r2g/scripts/kern_window.py $P/trt-$m-pf256.sqlite "measured pf256 r0" $P/trt-$m-pf256_window.csv > $P/trt-$m-pf256_window.txt 2>&1
done
# 2. the site's commit ee9098f60 at 120b pf256, a run and a kernel list
PIE_PY=/root/src/pie-ee9098/packages/python/server/python /root/r2g/scripts/pie-pb.sh openai/gpt-oss-120b 256 $B ee9098-120b-pf256-r1
PIE_PY=/root/src/pie-ee9098/packages/python/server/python /root/r2g/scripts/prof-pie-pb.sh openai/gpt-oss-120b 256 $P ee9098-120b-pf256
# 3. pie at 512 prompt tokens of its own rendering
for m in 20b 120b; do for n in 256 512; do
  /root/r2g/scripts/pie-pb.sh openai/gpt-oss-$m $n $E pie-$m-pf$n-r1 --prompts-file /root/prompts-gptoss-576.json
done; done
echo QUEUE2 DONE
