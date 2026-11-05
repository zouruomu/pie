#!/bin/bash
# usage: sweep.sh <outdir> <rounds> <engines...>   (engine: vllm | pie:<engine name in /root/engines>)
# Alternates engines within each round over $SHAPES.
out=$1; rounds=$2; shift 2
E=/root/r2g/pie/packages/python/server/python/pie/_engine.abi3.so
SHAPES=${SHAPES:-"c64 c128 c256 c512 c1024"}
for r in $(seq 1 $rounds); do
  for e in "$@"; do
    eng=${e%%:*}; tag=$eng
    if [ "$e" != "$eng" ]; then so=${e#*:}; cp /root/engines/$so.so $E; tag=pie-$so; fi
    for s in $SHAPES; do
      [ -e $out/$tag-$s-r$r.json ] && continue
      RES=$out /root/r2g/evidence/scripts/bench.sh $eng $s $tag-$s-r$r ${EXTRA:---dump-all-token-ids}
      nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader >> $out/$tag-$s-r$r.log
    done
  done
done
echo done > $out/sweep.done
