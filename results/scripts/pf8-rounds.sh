#!/bin/bash
# pf8 on cut B, base vs final, 3 alternating rounds: is the one-round -3% real? (A pf fire takes no decode step.)
SO=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
R=/root/r2g/evidence/results/raw/curve
M=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/mini60-e16
cd /root/r2g/pie/scripts/bench
for r in 1 2 3; do
  for e in base final; do
    cp /root/engines/$e.so $SO
    env -u CLAUDE_CODE_MESSAGING_TOKEN RUST_LOG=warn,runtime::engine::load=info PYTHONPATH=/root/r2g/pie/packages/python/client/src:/root/r2g/pie/packages/python/server/python \
      PIE_BENCH_INFERLET_DIR=/root/r2g/pie/examples/text-completion-bench \
      /root/venv-pie/bin/python pie_bench.py tput --model $M --prompts-file /root/prompts-512x80-b.json --max-tokens 1 --concurrency 8 --num-requests 16 \
      --warmup 2 --report-timing --dump-all-token-ids --json-out $R/b-$e-pf8-r$r.json > $R/b-$e-pf8-r$r.log 2>&1
    echo "EXIT=$? engine_sha256=$(sha256sum $SO | cut -c1-64)" >> $R/b-$e-pf8-r$r.log
  done
done
cp /root/engines/final.so $SO
