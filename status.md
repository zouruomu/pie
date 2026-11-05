# Status: 20261105-2035-a100-sxm-x1-gemma-4-31b-batching

**Phase:** setup / reproduce
**UTC:** 2026-11-05 21:03

- Pod up 20:37Z. Gemma-4-31B downloaded; vLLM 0.32.0 installed; pie engine (main e4bb499) building; inferlet built.
- Site rows read from benchmarks.pie-project.org/data/db.js: pie's best is c512 (latency p50 140 s ⇒ ~510 in flight), vLLM's best c1024 (latency p99 458 s ⇒ ~2,040 requests). Prompts: 169 tokens for batching (128 + Gemma's 41-token chat frame, from pbatch's 553 = 512 + 41).
- Running: vLLM c64–c1024 round 1 (pie build compiling beside it on the CPU).
- ETA early verdict: ~22:45Z. ETA hand-over: ~04:30Z.
- 20:49Z the pod's 300 GB disk filled (gpt-oss-120b's metal/ and original/ copies). Everything running died. Freed 60 GB; restarted at 21:00Z. Lost ~12 min.
- vLLM c64 r1: 601 tok/s, KV cache 11.1 GiB = 13,125 tokens (44 sequences of 297 tokens).
