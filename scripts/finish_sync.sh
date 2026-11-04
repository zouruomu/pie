#!/bin/bash
# Left running at hand-over: when the test processes given (pids) have exited, summarise the tests,
# run the secret scan and sync the evidence branch, retrying the push (GitHub returned transient 500s).
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
/root/r2g/evidence/scripts/tests_summary.sh
cd /root/r2g/evidence && bash scripts/secret_scan.sh || exit 1
for i in 1 2 3 4 5; do
  bash /root/r2g/sync.sh 2>&1 | grep -q "push failed" || break
  sleep 60
done
