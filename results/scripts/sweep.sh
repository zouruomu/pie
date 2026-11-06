#!/bin/bash
# usage: sweep.sh <outdir> <model q36|q38> <rounds> <engines...>   (engine: vllm | pie | pie:<python dir>)
# Alternates engines within each round over $SHAPES; skips results already present.
out=$1; m=$2; rounds=$3; shift 3
SHAPES=${SHAPES:-"c64 c128 c256 c512 c1024"}
for r in $(seq 1 $rounds); do
  for e in "$@"; do
    eng=${e%%:*}; tag=${TAG:-$eng}
    py=""; [ "$e" != "$eng" ] && { py=${e#*:}; tag=pie-$(basename $(dirname $py)); }
    for s in $SHAPES; do
      [ -e $out/$tag-$m-$s-r$r.json ] && continue
      PIE_PY=$py RES=$out /root/r2g/evidence/results/scripts/bench.sh $eng $m $s $tag-$m-$s-r$r ${EXTRA:---dump-all-token-ids}
    done
  done
done
echo done > $out/sweep-$m.done
