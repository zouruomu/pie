#!/bin/bash
# Teacher-forced logits on both cuts: base, fix, and base under the scalar selected walk
# (`sdpa.selected_mma = false` in a user profile layer, removed right after). Teacher: base's own greedy run.
P=/root/r2g/evidence/results/scripts/logit_probe.py
SO=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
U=/root/.pie/profiles/cuda/user/nvidia-b200.toml
A=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/mini60-e16
B=/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/mini60-e16
O=/root/probe/out
R=/root/r2g/evidence/results/raw
run() { # label engine model prompts teacher
  cp /root/engines/$2.so $SO
  env -u CLAUDE_CODE_MESSAGING_TOKEN RUST_LOG=warn,runtime::engine::load=info /root/venv-pie/bin/python $P $3 $4 $5 $O/$1 4 32 > $R/probe-$1.log 2>&1
  echo "EXIT=$? engine_sha256=$(sha256sum $SO | cut -c1-64)" >> $R/probe-$1.log
}
scalar() { mkdir -p $(dirname $U); printf 'api = 1\n\n[target]\ndevice = "NVIDIA B200"\n\n[kernel]\nsdpa.selected_mma = false\n' > $U; }
for cut in a b; do
  if [ $cut = a ]; then M=$A; PR=/root/prompts-512.json; else M=$B; PR=/root/prompts-512-m3.json; fi
  T=$R/ab-$cut-base-r1.json
  run $cut-base base $M $PR $T
  run $cut-fix fix $M $PR $T
  scalar; run $cut-scalar base $M $PR $T; rm -f $U
done
cp /root/engines/fix.so $SO
