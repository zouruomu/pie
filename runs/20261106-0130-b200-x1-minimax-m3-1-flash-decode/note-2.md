# Note 2 (orchestrator, 2026-11-06 03:12Z)

Early verdict accepted: I recomputed round 1 on cut B from results/raw (pie 9.621, vLLM 6.923, SGLang 7.593 ms ITL p50). Attributing and fixing on cut B, checked on cut A, is the right call. Points for rounds 2-3 and the report:

1. Same harness for every engine: pie ran `mode: latency`, vLLM and SGLang `mode: tput`. SGLang's TTFT p50 of 6.9 s (p95 13.1 s) means its 8 requests were all submitted at once and queued. ITL still looks like one sequence at a time (wall ≈ 8 × latency), but please run all three in the same mode at concurrency 1, or show in the report why the mode doesn't change ITL.
2. Cut B is MiniMax-M3 (not M3.1) at MXFP8. Note in the report any architectural difference from M3.1 that touches decode (config diff), so a reviewer can trust that a cut-B fix carries over to the site's model.
3. pie on cut B (9.62 ms) is slower than on cut A (8.98 ms). So the path to fix is most likely the MXFP8 expert/projection path at one row on sm_100. Keep the attribution table summing to the 2.7 ms delta against vLLM.
4. The identify bug (finding 4: an fp8 checkpoint identified as the mini's bf16 deployment and OOMing) is real and worth fixing, but keep it separate from the perf fix: write it up with `file:line` and a minimal repro. We'll decide later whether it ships as its own small PR.
