#!/bin/bash
# usage: curve.sh <label> — pie on cut A: throughput (256 in, 128 out; 128-token prompts are shorter than the chat template) at c4..c64 and prefill batch
# (512 in, 1 out) at pf4..pf32, whatever engine .so is installed. One JSON per point in results/raw/curve.
label=$1
R=/root/r2g/evidence/results/raw/curve; mkdir -p $R
M=${MODEL:-/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/mini60-e16}
so=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
cd /root/r2g/pie/scripts/bench
run() { # name prompts max_tokens concurrency num_requests
  env -u CLAUDE_CODE_MESSAGING_TOKEN RUST_LOG=warn,runtime::engine::load=info PYTHONPATH=/root/r2g/pie/packages/python/client/src:/root/r2g/pie/packages/python/server/python \
    PIE_BENCH_INFERLET_DIR=/root/r2g/pie/examples/text-completion-bench \
    /root/venv-pie/bin/python pie_bench.py tput --model $M --prompts-file $2 --max-tokens $3 --concurrency $4 --num-requests $5 \
    --warmup 2 --report-timing --dump-all-token-ids --json-out $R/$label-$1.json > $R/$label-$1.log 2>&1
  echo "EXIT=$? engine_sha256=$(sha256sum $so | cut -c1-64)" >> $R/$label-$1.log
}
for c in 2 4 8 16 32 64; do run c$c ${P256:-/root/prompts-256-a.json} 128 $c $((c * 4)); done
for n in 4 8 16 32; do run pf$n ${P512:-/root/prompts-512x64-a.json} 1 $n $((n * 2)); done
