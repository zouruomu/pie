#!/bin/bash
# alternating before/after with the two built engines (no user profile layer)
E=/root/src/pie/packages/python/server/python/pie/_engine.abi3.so
swap() { cp /root/engines/$1.so $E; }
rm -f /root/.pie/profiles/cuda/user/nvidia-geforce-rtx-4090.toml
for r in 1 2 3; do
  swap base-e6ed4df7f; /root/bench-pie.sh v-base-r$r --dump-all-token-ids
  swap fix;            /root/bench-pie.sh v-fix-r$r --dump-all-token-ids
  /root/bench-comp.sh sglang v-sgl-r$r
done
for r in 1 2; do
  for e in base-e6ed4df7f fix; do
    swap $e; t=${e%%-*}
    /root/bench-model.sh openai/gpt-oss-20b none v-oss20-$t-$r
    /root/bench-tput.sh openai/gpt-oss-20b none v-oss20-c256-$t-$r 256 512
    /root/bench-model.sh Qwen/Qwen3-0.6B none v-q06-$t-$r
    /root/bench-model.sh Qwen/Qwen3.5-4B none v-q35b4-$t-$r
  done
done
swap fix
echo done > /root/results/verify.done
