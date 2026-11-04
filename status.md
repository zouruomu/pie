# Status: 20261103-2218-l4-x1-gpt-oss-20b-batching
Phase: 3 (attribution). Updated 2026-11-04 00:25Z. Deadline ~2026-11-04 14:20Z.

Findings so far (round 1, current main 48a36f0d8, results/base):
- Site picks each engine's best point of the sweep (pie pbatch = pf8, vLLM = pf64; pie batching = c512, vLLM c1024 or c512), inferred from output_tok_s = N/wall and latency.
- pbatch: already green on main. pie best 7.67k (pf8) vs vLLM best 6.59k (pf128). Site's pie 4.67k was 8fb7e90e6.
- batching: pie best 1761 (c512) vs vLLM best 1790 (c512): -1.6% (1 round). Site's vLLM 2193 not reproduced here (its ITL/latency imply a faster card; the ratio is what transfers).
- Side: pie c16 291 vs vLLM 429 (-32%); profile knob moe.mxfp4_mma_per_from=1 gives c16 492 (+62%), c8 245->270.
- Side: pie pf512/pf1024 5.0-5.3k (1024-row fires once max_forward_requests >= 512) vs vLLM 6.5k.

Running: knob A/B (per_from, working_set 50%), then the measured commit 8fb7e90e6 on this pod (results/old).
Downloading: mimo-v2.6 mini (layers 0-5), gpt-oss-120b mini (layers 0-3) for cross-model checks.
