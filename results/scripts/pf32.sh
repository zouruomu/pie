#!/bin/bash
# pf32 on both cuts, base vs final (the curve's 64-prompt files were 2 short of 32 x 2 + 2 warmups).
S=/root/r2g/evidence/results/scripts
SO=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
R=/root/r2g/evidence/results/raw/curve
cd /root/r2g/pie/scripts/bench
for cut in a b; do
  if [ $cut = a ]; then M=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/mini60-e16; else M=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/mini60-e16; fi
  for e in base final; do
    cp /root/engines/$e.so $SO
    env -u CLAUDE_CODE_MESSAGING_TOKEN RUST_LOG=warn,runtime::engine::load=info PYTHONPATH=/root/r2g/pie/packages/python/client/src:/root/r2g/pie/packages/python/server/python \
      PIE_BENCH_INFERLET_DIR=/root/r2g/pie/examples/text-completion-bench \
      /root/venv-pie/bin/python pie_bench.py tput --model $M --prompts-file /root/prompts-512x80-$cut.json --max-tokens 1 --concurrency 32 --num-requests 64 \
      --warmup 2 --report-timing --dump-all-token-ids --json-out $R/$cut-$e-pf32.json > $R/$cut-$e-pf32.log 2>&1
    echo "EXIT=$? engine_sha256=$(sha256sum $SO | cut -c1-64)" >> $R/$cut-$e-pf32.log
  done
done
cp /root/engines/final.so $SO
