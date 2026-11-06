#!/bin/bash
# usage: trt.sh <q36|q38> <label> [extra trtllm_tput.py args...]
m=$1; label=$2; shift 2
case $m in
  q36) M=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/*); PF=/root/prompts/q36-c167.json ;;
  q38) M=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*); PF=/root/prompts/q38-c205.json ;;
esac
OUT=${OUT:-/root/r2g/evidence/results/trt}
mkdir -p $OUT
cd /root
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader -lms 1000 > $OUT/$label.clk 2>/dev/null &
CLK=$!
env -u CLAUDE_CODE_MESSAGING_TOKEN TLLM_LOG_LEVEL=INFO timeout 5400 /root/venv-trtllm/bin/python -I /root/r2g/evidence/results/scripts/trtllm_tput.py \
  --model $M --prompts-file $PF --json-out $OUT/$label.json "$@" > $OUT/$label.log 2>&1
echo EXIT=$? >> $OUT/$label.log
kill $CLK
