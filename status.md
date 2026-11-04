# Status: 20261103-2218-l4-x1-gpt-oss-20b-batching
Phase: 5 (verification). Updated 2026-11-04 03:00Z. Deadline ~2026-11-04 14:20Z.

Findings so far (current main 48a36f0d8 unless said):
- Site picks each engine's best point of the sweep (pie pbatch = pf8, vLLM pf64; pie batching c512, vLLM c512/c1024).
- pbatch: green on main (pie best 7.67k pf8 vs vLLM best 6.59k pf128). The measured commit 8fb7e90e6 on this pod also gives pf8 7.3k: the site's 4.67k does not reproduce.
- batching: pie best 1761-1809 (c512) vs vLLM 1790 (c512). Measured commit 8fb7e90e6 on this pod: best 1684 (c128), c512 1436.
- #1368's parent (ee9098f60): c256 1980/1994 (2 runs) vs main 1718-1742: the #1368 admission cap (5 x declared-context seqs = 215 seats at c256) splits the harness's 512 lockstep requests into 3 waves (215/215/82).
- c16: pie 300 vs vLLM 429; moe.mxfp4_mma_per_from 4 -> 0 gives ~490 (grouped wmma -> run kernel, nsys in results/prof). Commit 003d8ba06 on the fix branch (not pushed yet).

Running: pool/admission knobs at c256/c512 (results/knobs2). Next: decide fix lines, then base/fix/vLLM rounds, cross-model (gpt-oss-120b mini, mimo mini), tests.

03:00Z update:
- Fix decided: L4 profile moe.mxfp4_mma_per_from 4 -> 0 (commit 003d8ba06, engine sha 4662a3bd). Mechanism: c8/c16 decode selects (32/64 routes over 32 experts) ran the by-route mxfp4 select (`moe_matmul_select_mxfp4`, reads an expert once a route), 41 ms of a 54 ms step at c16; the run kernel 22.5 ms.
- No profile knob recovers c256 (admission cap): rec512 breaks graph arming, seated_first=false c256 +15% but c512 -11%, seat_to_end/ws55 no gain.
- gptoss-20b-mini (5 layers, 16 experts): c8 794 -> 888, rest flat; raw/pf8 tokens identical; concurrency outputs are nondeterministic run to run even on one engine (noise floor measured).
- gpt-oss-120b miniature does not fit an L4 (pie's mini keeps 36 layers); mimo all-expert mini does not fit (21.1 GB weights); running mimo 6 layers x 32 experts instead.
Running: #1350 isolation (old engine + ws65), mimo e32 base/fix, then verification sweep (fix r1; base/fix/vLLM r2-r3).
