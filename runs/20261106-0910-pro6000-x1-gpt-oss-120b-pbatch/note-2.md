# Note 2 (orchestrator, 09:42Z)

Good early findings. Direction for the next hours:
1. Split the gap in your attribution: (a) what the MoE precision accounts for (W4A16 vs TRT-LLM's W4A8), and (b) everything else: attention, the dense projections, norms, host/fire overhead. Measure it, from the nsys per-component table at pf256, not from estimates. A (b) gap that pie can close at its current precision is the cleanest fix to ship.
2. A W4A8 gpt-oss deployment changes numerics and the release's precision. That is a maintainer decision, so it ships only with hard quality evidence: teacher-forced KL / top-1 agreement against the W4A16 path over a few hundred tokens, greedy-token agreement on raw and batching runs, and ideally a short task-accuracy check. It also has to be clear what the site would then measure (a declared deployment the harness picks, or opt-in only). If the sm_120a kernel needs K % 128 == 0 and gpt-oss has K = 2880, give a size estimate (hours) for handling the 64-wide tail before starting it.
3. The 448 vs 512 prompt tokens is an important harness finding: say per engine how many tokens it actually prefilled, and compare tok/s on equal token counts too.
