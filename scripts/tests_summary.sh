#!/bin/bash
# Summarise every test log in results/tests (passed / failed / last result line) into results/tests/SUMMARY.txt.
cd /root/r2g/evidence/results/tests
{
  date -u +"as of %Y-%m-%dT%H:%MZ"
  for f in *.log; do
    printf '%-38s ok=%-4s failed=%-3s %s\n' "$f" "$(grep -c '\.\.\. ok' $f)" "$(grep -c '\.\.\. FAILED' $f)" \
      "$(grep -E '^test result|EXIT' $f | tr '\n' ' ' | cut -c1-160)"
  done
} > SUMMARY.txt
