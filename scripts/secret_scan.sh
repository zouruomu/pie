#!/bin/bash
# Fail if any secret value from the environment appears in the evidence tree (prints names only).
rc=0
for v in CLAUDE_CODE_OAUTH_TOKEN R2G_GH_TOKEN RUNPOD_API_KEY HF_TOKEN GH_TOKEN GITHUB_TOKEN; do
  val="${!v}"; [ -z "$val" ] && continue
  if grep -rIlaF -- "$val" /root/r2g/evidence --exclude-dir=.git >/dev/null 2>&1 || grep -rlaF -- "$val" /root/r2g/evidence/results >/dev/null 2>&1; then echo "FOUND $v"; rc=1; fi
done
[ $rc = 0 ] && echo "secret scan clean"
exit $rc
