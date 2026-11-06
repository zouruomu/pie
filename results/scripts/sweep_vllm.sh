#!/bin/bash
# vLLM 0.32.0 over the pf sweep (W4A16 reference), one point a process, as pie_bench runs pie.
R=/root/r2g/evidence/results/baseline
[ -z "$NO_Q3" ] && /root/r2g/scripts/queue3.sh > /root/r2g/logs/queue3.log 2>&1
cd /root/r2g/pie/scripts/bench
for m in ${MODELS:-20b 120b}; do for n in ${POINTS:-4 8 16 32 64 128 256 512 1024}; do
  l=vllm-$m-pf$n-r${ROUND:-1}
  [ -f $R/$l.json ] && continue
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader > $R/$l.clocks
  env -u CLAUDE_CODE_MESSAGING_TOKEN /root/venv-vllm/bin/python vllm_bench.py tput --model openai/gpt-oss-$m \
    --num-requests $n --concurrency $n --warmup 16 --max-tokens 1 --prompts-file /root/prompts-gptoss-512.json \
    --json-out $R/$l.json > $R/$l.log 2>&1
  echo EXIT=$? >> $R/$l.log
done; done
echo VLLM SWEEP DONE
