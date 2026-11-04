# Status: 20261103-2218-l4-x1-gpt-oss-20b-batching

Phase: 7 (write-up), awaiting the last kernel tests. Updated 2026-11-04 11:20Z.

- Fix: `fa52aea7` on `r2g/20261103-2218-l4-x1-gpt-oss-20b-batching`, rebased on upstream `59935820d`.
  One L4 profile line, `moe.mxfp4_mma_per_from` 4 -> 0.
- Report: report.md. PR text: pr.md.
- Self-reviews: two independent subagent reviews (code claims, numbers) done; their corrections are
  applied.
- Remaining: kernels-cuda moe_select tests (running) and the engine-cuda GPU suite (20/60 on each
  tree, all passing, about 25 an hour). `scripts/finish_sync.sh` syncs this branch when the suites
  exit.
