# Outcome: 20261103-2218-l4-x1-gpt-oss-20b-batching

**Verdict:** no PR. Both target cells are already green on current pie main, so the rule "gap already green" applies. The pod also found and verified a real fix for a red stretch of the batching curve below the site's best point (c8/c16). It is left on `r2g/20261103-2218-l4-x1-gpt-oss-20b-batching` (fa52aea7, on upstream 59935820d) with pr.md ready, for Felix to decide.

## Cells (NVIDIA L4, gpt-oss-20b mxfp4, vs vLLM 0.32.0)

| cell | site | on this pod, current main (orchestrator recomputed from results/verify + verify2) |
|---|---|---|
| pbatch (best pf-N) | pie 4,666 vs 6,425 (-27.4%) | pie ~7.6k (pf4-pf256 7.2-7.7k) vs vLLM 6,576 (pf256): green, about +16% |
| batching (best c-N) | pie 1,576 vs 2,193 (-28.1%) | pie best 1,743 (c128) / 1,746 (c512, fix build) vs vLLM 1,782 (c512): -2.2%, not red |

- The site's pie values don't reproduce. The measured commit 8fb7e90e6, rebuilt on the pod, gives pf8 7,325.
- The site's vLLM batching value (2,193) is about 23% above what vLLM 0.32.0 does on this card.

## Fix found (not shipped)

- **Change:** one line in the L4 profile, `moe.mxfp4_mma_per_from` from 4 to 0.
- **Why it helps:** c8/c16 decodes (1-2 routes an expert) ran the by-route MXFP4 select, which streams an expert's codes once per route: 40 ms of a 50 ms c16 step. With the line, they take the 512-code run kernel that c25-c128 already use.
- **Recomputed, median of 2-3 rounds:**
  - c16: 302 -> 495 tok/s (vLLM 448).
  - c8: 240 -> 274 (vLLM 277).
  - c4: 169 -> 174.
  - Every other point from c32 to c1024, pf4 to pf1024, and raw is within the run spread. pf8 is bimodal with clocks on this 72 W card.
  - The engine sha256 matches per build (base 07db7e20, fix 4662a3bd).
- **Cross-model:** gptoss-20b-mini c8 +12%, MiMo 32-expert mini c8 +37%. Nothing slower beyond spread, except c24 -2.2% (2 rounds).
- **Tokens:** greedy tokens are identical base vs fix for raw and pbatch on every model. At c8/c16, the outputs differ about as much as pie differs from itself run to run, since concurrency batching is nondeterministic.
- **Tests:**
  - Profile-resolution tests pass on both builds.
  - The engine-cuda GPU suite was 20/60 on each tree, all passing, when the run was closed. It runs at about 25 tests an hour, which can't finish before the deadline.
  - kernels-cuda moe_select: 5 passed on both builds, and `the_routed_selects_answer_the_routed_dot` failed identically on both. (Corrected 2026-11-05: the pod's final report had this right. The follow-up run traced the failure to #1274's W4A8 case, a pre-existing test bug, not the pipe kernel.)

## Side findings

- **#1368 admission cap:** it costs the L4 c256 13% (1,987 -> 1,729). It splits the 512 lockstep requests into 3 waves. Without it, pie's c256 beats vLLM's best by 11%. This is engine-wide, so worth telling Anton.
- **pf512/pf1024:** pie gets 5.0-5.3k vs vLLM 6.5k (-23%), because 1024-row fires start once max_forward_requests >= 512. This is not a site cell, since the site takes each engine's best point.
- **#1350 (L4 65% working set):** it already lifted pbatch and batching after the site measured.

## Cost

- Pod wkwrx6vogbaev0: NVIDIA L4, $0.59/h, US-MO-2.
- Ran 2026-11-03 22:19Z to 2026-11-04 11:58Z, 13.6 h, about $8.05.
- The orchestrator deleted the pod (DELETE 204, then GET 404 at 11:58Z). Its agent was still running the GPU suite, so the bootstrap never reached its done check. Evidence after the 11:36Z sync was not pushed.
- No review rounds were needed for the verdict.
