#!/bin/bash
until grep -q 'VLLM SWEEP DONE' /root/r2g/logs/vllm-sweep.log 2>/dev/null; do sleep 10; done
source /root/r2g/scripts/paths.sh
S=/root/r2g/scripts/pie-pb.sh
X=/root/r2g/evidence/results/explore
EA=/root/src/pie-ea800e/packages/python/server/python
# 1. ea800efeb (the site's 20b commit) vs main at the small points, alternating
for r in 1 2; do for n in 8 16 32 64; do
  PIE_PY=$EA $S openai/gpt-oss-20b $n $X ea800e-20b-pf$n-r$r
  $S openai/gpt-oss-20b $n $X main-20b-pf$n-r$r
done; done
# 2. fire split: pf16/pf32 with 8192-row fires instead of the profile's 5120
for r in 1 2; do for m in 20b 120b; do for n in 16 32; do
  $S openai/gpt-oss-$m $n $X f8192-$m-pf$n-r$r --max-forward-tokens 8192
  $S openai/gpt-oss-$m $n $X f5120-$m-pf$n-r$r
done; done; done
echo QUEUE4 DONE
