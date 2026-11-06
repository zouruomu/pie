#!/bin/bash
# review 1: ea800efeb vs main at 20b pf32/pf256/pf1024, 3 alternating rounds
S=/root/r2g/scripts/pie-pb.sh
X=/root/r2g/evidence/results/ea800e-ab
EA=/root/src/pie-ea800e/packages/python/server/python
for r in 1 2 3; do for n in 32 256 1024; do
  PIE_PY=$EA $S openai/gpt-oss-20b $n $X ea800e-20b-pf$n-r$r
  $S openai/gpt-oss-20b $n $X main-20b-pf$n-r$r
done; done
echo EA_AB DONE
