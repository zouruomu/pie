# Note 3 (orchestrator, 2026-11-06 04:02Z)

Fix 1 looks like the right direction. Since it changes greedy tokens on 6/8 requests, "a near-tie miniature" needs numbers before In Gim will accept it:
1. At the first diverging step of each request, give the base's top-2 logit margin (show it is a near-tie), and over a fixed teacher-forced sequence give the max abs logit difference and KL of fix vs base. Put next to them the same metrics for a reduction-order change that is accepted already (e.g. base vs base at a different `selected` warp count, 16 vs 8, or the bf16-GEMV experiment you ran). Run it on cut A and cut B.
2. Show the cluster path is off where it shouldn't run: batching c32+ and pbatch on cut B (one round each, base vs fix: same numbers within spread and same kernel per nsys, or the decision rule's log).
3. Keep fix 2 (q/k norm + RoPE) a separate commit, measured on its own, so we can ship them as one PR or two.
