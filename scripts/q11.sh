#!/bin/bash
# After q8: fix v3's round-2 points that ran on the wrong engine (results/bad) run again.
D=/root/r2g/evidence/scripts; R=/root/r2g/evidence/results
until [ -e $R/q8.done ]; do sleep 20; done
SHAPES="c128 c256 c512 c1024" $D/sweep.sh $R/base 2 pie:fix-v3
echo done > $R/q11.done
