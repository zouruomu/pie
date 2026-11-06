#!/bin/bash
# usage: bench.sh pie|vllm|sglang <label> [extra bench args...]
# Raw workload (concurrency 1, 512-token chat-rendered prompts, 256 out) on the mini60-e16 cut.
eng=$1; label=$2; shift 2
R=/root/r2g/evidence/results/raw; mkdir -p $R
M=${MODEL:-/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/mini60-e16}
P=${PROMPTS:-/root/prompts-512.json}
cd /root/r2g/pie/scripts/bench
so=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
{ echo "date: $(date -u +%FT%TZ)"; echo "engine: $eng"; [ $eng = pie ] && echo "engine_sha256: $(sha256sum $so | cut -d' ' -f1)";
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader; } > $R/$label.meta
if [ $eng = pie ]; then
  env -u CLAUDE_CODE_MESSAGING_TOKEN PYTHONPATH=/root/r2g/pie/packages/python/client/src:/root/r2g/pie/packages/python/server/python \
    PIE_BENCH_INFERLET_DIR=/root/r2g/pie/examples/text-completion-bench \
    /root/venv-pie/bin/python pie_bench.py latency --model $M --requests ${REQ:-8} --warmup 2 --max-tokens 256 \
    --prompts-file $P --report-timing --dump-all-token-ids --json-out $R/$label.json "$@" > $R/$label.log 2>&1
else
  env -u CLAUDE_CODE_MESSAGING_TOKEN /root/venv-$eng/bin/python ${eng}_bench.py latency --model $M --requests ${REQ:-8} --warmup 2 \
    --max-tokens 256 --prompts-file $P --report-timing --dump-all-token-ids --json-out $R/$label.json "$@" > $R/$label.log 2>&1
fi
echo EXIT=$? >> $R/$label.log
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader >> $R/$label.meta
