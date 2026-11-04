#!/bin/bash
# Left running at hand-over: when both engine-cuda suites have exited, summarise the tests, run the
# secret scan and sync the evidence branch.
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
/root/r2g/evidence/scripts/tests_summary.sh
cd /root/r2g/evidence && bash scripts/secret_scan.sh && bash /root/r2g/sync.sh
