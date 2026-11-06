# Status: 20261105-2035-a100-sxm-x1-gemma-4-31b-batching

**READY FOR REVIEW**

**Phase:** review 1 answered
**UTC:** 2026-11-06 11:42

## Verdict
- **The gap is real on current main, and larger than the site's:** pie's best is 474.8 tok/s (c256) against vLLM 0.32.0's 656.5 (c128), -27.7% (site: -18.1%).
- **The cause is seating, not the step.** pie decodes ~21 sequences at once (46 ms a token) against vLLM's 31-35 (52 ms).
  - #1368's admission cap admits 5 × pool / 2048 = 26 at c512, where the pool seats 35.
  - At the default 50% working set the pool fit leaves ~0.5 GiB of its room unused.

## Fix
- Commit `54385f76` on upstream main `dc6282d7`, pushed as origin `r2g/20261105-2035-a100-sxm-x1-gemma-4-31b-batching`. 10 files, +57/-2 (`fix.patch`).
- It is three A100 profile lines and one new profile decision, `engine.uncapped_admission` (default off, so no other card changes):
  1. `seat_to_end = true`;
  2. `uncapped_admission = true`, which lifts the cap only while the pools seat to the end;
  3. `working_set_percent = 40`.
- Note 1 was followed: the shared runtime change in v1 was narrowed to a profile decision.

## Results (report.md has everything; pr.md has the PR text)
- **Gemma-4-31B c64-c1024** (3 rounds; 2 for the fix at c128-c1024): +18.5% to +31.4%.
  - Best 474.8 -> 598.6, -8.8% against vLLM's best, level at c1024.
  - c32 +37%; c4-c16 and raw unchanged; pbatch best 3735 -> 3829.
  - 1024 tokens out at c128: 148 -> 217.
  - Final rebased A/B: c128 457.7 -> 592.5, c512 454.1 -> 597.2.
- **Cross-model:** Gemma-4-26B c512 +10.7%, c1024 +2.1%; gpt-oss-120b +0.6 / +1.8 / +9.1% (c256/c512/c1024); Qwen3-0.6B unchanged.
- **Costs:**
  - ITL p99 at Gemma-4-31B c1024: 49 -> 481 ms (vLLM 649).
  - **Gemma-4-26B c512 (Review 1 item 1, traced): the fix suspends lanes in 3 of 5 runs** (ITL p99 430-680 ms; the 2 clean runs 89-90; main 177-360 over 3 runs).
    - Uncapped admission seats all 512 before any process finishes, and seat_to_end's gate has no footprint until one does. 512 × 19 pages > the 5,887-page pool (it carries 309 to the end).
    - In stalled runs 17-18 processes are paused (KV copied to host and back, no replay), page allocations queue, and lanes stall in cohorts for 0.6-2.7 s.
    - Main's stalls (~360 ms, fire length) have 0 pauses.
    - Throughput (4793-5162 against 4677-4868) and TTFT p50 (2.3-3.1 s against 6.6-7.1 s) stay better in every run.
    - Proposed remedy, not done: bound the first wave until a footprint is learned. This decides whether it ships; report.md §7.
  - TTFT falls everywhere on the target cell.
- **Tokens:** identical base vs fix where the cap does not bind (raw, c8, c16, pf4-pf512; 1 of 8 c4 requests differs). Above it the lanes a step differ; agreement with vLLM's own ids is base 25.9-26.2%, fix 25.2-25.5%.
- **Tests on 54385f76:**
  - runtime 101, pie-profile+worker+engine 78, engine-cuda chip 13, metal/vulkan check: all pass.
  - engine-cuda lib 69/69.
  - GPU integration 58 passed, 1 ignored, 1 failed; ep 5 of 6. Both failures are identical on base e4bb499 (pre-existing).
  - 8 integration tests failed while the disk was full and passed on rerun. a_joined_projection panics at :245/:309 on both builds; the EP test at :358 on both.

## Notes and slips (all in report.md)
- The disk filled twice: gpt-oss-120b's extra copies, then the engine-cuda tests' /tmp checkpoints. Freed both times; 126 GB is free now.
- Two runs were invalidated and moved to results/bad/: a bench.sh edited mid-run, and an engine left behind by a slotted run. Both were caught by the sha256 check.
- A fresh-context self-review ran twice. Its findings were fixed, including a logic gap in the first bootstrap filter.
- Side findings:
  - the pool fit strands ~0.5 GiB when its first pass keeps coarse map units (store.rs);
  - the A100 gpt-oss-120b batching cell is green on current main;
  - the H100 Gemma-4-31B cell is probably the same cap (not measured).
- Note 1 (23:10Z) acknowledged above.
