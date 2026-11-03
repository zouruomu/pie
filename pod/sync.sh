#!/bin/bash
# Push /root/r2g/evidence, with the bootstrap log and agent results, to r2g-evidence/<run>.
R=/root/r2g
cp "$R/bootstrap.log" "$R"/agent-*.json "$R/evidence/" 2> /dev/null
git -C "$R/evidence" add -A
git -C "$R/evidence" commit -q -m "r2g ${R2G_RUN:-run} evidence" || true
git -C "$R/evidence" push -q -f origin HEAD || echo "evidence push failed"
