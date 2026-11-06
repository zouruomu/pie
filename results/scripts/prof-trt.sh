#!/bin/bash
# usage: prof-trt.sh <q36|q38> <cN> <label> [extra trtllm_tput.py args...]
# nsys over the TRT-LLM process tree; kernels are later cut to the NVTX range "measured cN r0".
m=$1; shape=$2; label=$3; shift 3
case $m in
  q36) M=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/*) ;;
  q38) M=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*) ;;
esac
OUT=/root/r2g/evidence/results/prof; mkdir -p $OUT
cd /root
env -u CLAUDE_CODE_MESSAGING_TOKEN /usr/local/cuda/bin/nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=node \
  --trace-fork-before-exec=true -f true -o /root/prof/$label \
/root/venv-trtllm/bin/python -I /root/r2g/evidence/results/scripts/trtllm_tput.py --model $M --prompts-file /root/prompts/site-c128.json \
  --points ${shape#c} --max-batch-size 128 --json-out $OUT/$label.json "$@" > $OUT/$label.log 2>&1
echo EXIT=$? >> $OUT/$label.log
/usr/local/cuda/bin/nsys export -t sqlite -f true -o /root/prof/$label.sqlite /root/prof/$label.nsys-rep > /dev/null 2>&1
