#!/bin/bash
# wrap-up queue (note 4), one engine at a time
while pgrep -f 'pie_bench.py|vllm_bench.py|trtllm_pbatch.py' > /dev/null; do sleep 5; done
source /root/r2g/scripts/paths.sh
S=/root/r2g/scripts/pie-pb.sh
X=/root/r2g/evidence/results/explore
B=/root/r2g/evidence/results/baseline
EA=/root/src/pie-ea800e/packages/python/server/python
# a. the 20b pf16 dip: 8192-row fires vs the profile's 5120, and ea800efeb vs main
for n in 16 32; do
  $S openai/gpt-oss-20b $n $X f8192-20b-pf$n-r1 --max-forward-tokens 8192
  $S openai/gpt-oss-20b $n $X main-20b-pf$n-r1
  PIE_PY=$EA $S openai/gpt-oss-20b $n $X ea800e-20b-pf$n-r1
done
$S openai/gpt-oss-120b 16 $X f8192-120b-pf16-r1 --max-forward-tokens 8192
# b. pie vs vLLM at the points within 4.5%: rounds 2-3, alternating
for r in 2 3; do
  for p in "120b 64" "120b 256" "120b 1024" "20b 512"; do
    set -- $p
    MODELS=$1 POINTS=$2 ROUND=$r NO_Q3=1 /root/r2g/scripts/sweep_vllm.sh > /dev/null 2>&1
    [ -f $B/pie-$1-pf$2-r$r.json ] && l=pie-$1-pf$2-r${r}v || l=pie-$1-pf$2-r$r
    $S openai/gpt-oss-$1 $2 $B $l
  done
done
echo QUEUE7-AB DONE
# c. 120b batching at c1024 (128 in, 128 out): pie (its own 128-token rendering) and TRT-LLM, one each
BT=/root/r2g/evidence/results/batching
mkdir -p $BT
cd /root/r2g/pie/scripts/bench
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader > $BT/pie-120b-c1024.clocks
env -u CLAUDE_CODE_MESSAGING_TOKEN PYTHONPATH=/root/r2g/pie/packages/python/server/python:/root/r2g/pie/packages/python/client/src \
  PIE_BENCH_INFERLET_DIR=/root/r2g/pie/examples/text-completion-bench \
  /root/venv-pie/bin/python pie_bench.py tput --model openai/gpt-oss-120b --num-requests 1024 --concurrency 1024 --warmup 16 \
  --max-tokens 128 --prompts-file /root/prompts-gptoss-192.json --json-out $BT/pie-120b-c1024.json > $BT/pie-120b-c1024.log 2>&1
(cd /tmp && env -u CLAUDE_CODE_MESSAGING_TOKEN /root/venv-trtllm/bin/python -I /root/r2g/scripts/trtllm_pbatch.py --model $M120 \
  --points 1024 --max-tokens 128 --prompts-file /root/prompts-gptoss-128.json --json-out $BT/trt-120b-c1024.json > $BT/trt-120b-c1024.log 2>&1)
# d. pie's own sm_120a W4A8 select at gpt-oss-120b's shape (K padded to 3072), an existing kernel timed
W=/root/r2g/evidence/results/w4a8
mkdir -p $W
cd /root/src/pie-w4a8
env -u CLAUDE_CODE_MESSAGING_TOKEN W4A8_TMA=1 W4A8_WS=1 /usr/local/cuda/bin/nsys profile -t cuda -s none --cpuctxsw=none -f true -o $W/w4a8-gptoss-timing \
  /root/target-w4a8/release/deps/kernels_cuda-84de8ca1e1204af8 --ignored --exact linear::moe_select::w4a8_select_timing_gptoss --test-threads=1 > $W/w4a8-gptoss-timing.log 2>&1
echo EXIT=$? >> $W/w4a8-gptoss-timing.log
/usr/local/cuda/bin/nsys stats -q --report cuda_gpu_kern_sum --format csv -o $W/w4a8-gptoss-timing $W/w4a8-gptoss-timing.nsys-rep > /dev/null 2>&1
echo QUEUE7 DONE
