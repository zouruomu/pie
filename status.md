# Status: 20261106-0130-b200-x1-minimax-m3-1-flash-decode

UTC: 2026-11-06 03:25Z
Phase: attribution done (first pass); building the first fix (selected attention split over a cluster).

## Early verdict (B200, main e2e9d1e, SM 1965 MHz)
The gap is real, but not as the site states it.
| cut | engine | ITL p50 ms | decode tok/s (1000/ITL) | rounds |
|---|---|---|---|---|
| A: MiniMax-M3.1-Flash mini60-e16 (MXFP8 proj + NVFP4 experts; the site's base_model) | pie (MX8+NVFP4 deployment) | 9.01 / 8.98 / 8.97 | 111.2 | 3 |
| A | vLLM 0.32.0, 0.33.0; SGLang 0.5.22 | cannot load (NVFP4 expert scales read as MXFP8) | - | - |
| B: MiniMax-M3-MXFP8 mini60-e16 (same architecture, MXFP8 everywhere) | pie (MX8 deployment) | 9.62 | 104 | 1 (2 more running) |
| B | vLLM 0.32.0 | 6.92 / 6.88 / 6.87 | 145.4 | 3 |
| B | SGLang 0.5.22 | 7.59 / 7.52 / 7.52 | 133.0 | 3 |
On cut B pie trails vLLM by 28.5% and SGLang by 21.8% (decode tok/s = 1000/ITL p50, medians). vLLM/SGLang ITL from the streaming c1 client (tput --concurrency 1 --report-timing); pie from its latency client. Site: pie 8.99, vLLM 5.73, SGLang 7.80.
Whatever the site's vLLM ran (most likely a dev build or a different cut), cut B is the fair same-weights comparison I can reproduce, so I attribute and fix on it and check every change on cut A too.

## Per-step attribution, cut B (nsys node traces; results/analysis/components-*.tsv, scripts/components.py)
Exposed time per component (each kernel charged end minus the latest earlier end on any stream), us a step:
| component | pie (9640) | vLLM (6845) | delta |
|---|---|---|---|
| GEMMs: projections + routed experts | 4117 | 2649 | +1468 |
| norms, q/k norm + RoPE, KV append | 1412 | 602 | +810 |
| attention (sparse + 3 dense layers) | 1092 | 409 | +683 |
| routing: gate, top-k, layout, gather, sum | 1106 | 589 | +517 |
| activation quantize (e4m3 1x32) | 863 | 461 | +402 |
| SwiGLU | 271 | 190 | +81 |
| index top-k | 102 | 243 | -141 |
| lm_head | 350 | 404 | -55 |
| other | 37 | 78 | -41 |
| gaps (no kernel running) | 290 | 1220 | -930 |
Per MoE layer on the critical path (pie 155 us vs vLLM ~108): q/k norm + RoPE + KV append +12 (two single-CTA fused row skeletons, 14.4 us, vs vLLM's one 10-CTA fused kernel, 2.3); sparse attention +11.6 (pie's selected MMA kernel runs 4 CTAs, one a kv head, 18.4 us; vLLM splits 16 ways a head, 5.0 + 2.2 merge); expert glue between the gate-up and down GEMMs +11 (gather, SwiGLU, layout, quantize); qkv projection +6; expert down GEMM +5.5; routing layout +4.
The attention and q/k-norm causes are the same on cut A (the site's checkpoint): selected attention 18.4-18.6 us a layer there too.

## Findings so far (raw logs in results/raw, results/logs)
1. Cut: MiniMaxAI/MiniMax-M3.1-Flash @ce7f4df8 (weights unchanged since 10-23), layers 0-59, 16 of 128 experts: 43.41 GiB, MXFP8 projections + NVFP4 experts. Site DB row (data/tags/b200-full.json) confirms mini_k 0.125, 60 layers, base_model MiniMaxAI/MiniMax-M3.1-Flash.
2. Params (note 1 / note-1 of the last run): my cut counts 64.67 B text params (routed 51.64, other 7.34, shared 3.23, embedding 2.458 B, exactly the site's), active 25.9 B; vision 0.87 B and scales 3.56 B apart (results/param_count.txt). Site: 124.25 B / 23.0 B active. 124.25 matches no integer expert count (it would need ~35 of 128); the embedding matches exactly and mini_k 0.125 = 16/128, so I read 124.25 as the site's formula, not a different cut. For c=1 decode it can't matter: top-4 + shared are read per layer either way.
3. **vLLM 0.32.0 and SGLang 0.5.22 cannot load this checkpoint** (both: "size of tensor a (192) must match b (384)": they read the NVFP4 experts' K/16 scales as MXFP8 K/32 scales). vLLM's own recipe says M3.1-Flash needs vLLM >= 0.33.0 (released 11-02, the day before the site's "0.32.0" vLLM run). So the site's vLLM/SGLang numbers were not measured on this checkpoint at those versions, or the versions are mislabelled.
4. **pie auto-identifies this cut as bf16** (minimax-m3-mini-bf16-kv-bf16-l60-e16): identify (crates/runtime/src/engine/load.rs:267) takes the first deployment that compiles, and models/minimax_m3.poem lists the mini's BF deployment before [MX8, nvfp4a4]. It widens to 124.5 GB on the GPU and OOMs this pod's 264 GiB cgroup on the host. --precision can't override it (refused after identify). With the mini's [MX8, nvfp4a4] listed first (an edit in ~/.pie/models only, to be reverted), pie serves it at 44 GB: **ITL p50 9.01 ms, 110 tok/s** (1 round), matching the site's 8.99.

5. vLLM 0.33.0 fails identically. vLLM's recipe and SGLang's cookbook both serve M3.1-Flash only from dev builds (vllm/vllm-openai:minimax-m31-flash; lmsysorg/sglang:dev-minimax-m31-flash, "support not in main yet"). **No released vLLM or SGLang loads this checkpoint**, so the site's 174.43 / 128.24 can't be this checkpoint at 0.32.0 / 0.5.22.
6. pie on the M3.1-Flash cut, 3 rounds: ITL p50 9.01 / 8.98 / 8.97 ms (SM 1965 MHz, max). nsys: 9.02 ms/step, 1649 kernels, 43 us idle. MoE layer ~146 us: qkv proj 18, q/k norm+rope (two 1-CTA kernels) 10, selected attention 18.6, o_proj 14, NVFP4 route+quant 11, routed experts W4A4 43, rest small. Weight bytes alone ~45 us/layer at 6.5 TB/s. Tried MXFP8 one-row projections on the existing bf16 GEMV (fp8_block_gemv takes 32-wide blocks): 8.98 -> 8.87 ms only, tokens diverge (W8A16 vs W8A8); dropped.

## Running
- Cut of MiniMaxAI/MiniMax-M3-MXFP8 (same architecture, all-MXFP8 experts: "the MXFP8 release", the site's scheme wording and what 0.32.0/0.5.22 can load), 60 layers, 16 experts -> all three engines on it.

Blocker: none, but the cell's baseline is ambiguous; I will report both cuts.
ETA early verdict: ~03:30Z. Hand-over target: ~11:15Z.
Orchestrator note 2 (03:12Z): acknowledged.
1. Mode: pie in the same streaming tput c1 mode on cut B gives ITL 9.66 ms (results/raw/pie-m3-tput-r1) against 9.62 in latency mode; SM clock 1965 MHz in-run. SGLang's TTFT is queueing (its client submits all 8 at t=0); its ITL is one sequence at a time. I'll quote TTFT only from c1 runs where requests don't queue.
2. Config diff M3-MXFP8 vs M3.1-Flash (results/analysis/config-diff-m3-vs-m31.txt): same shapes; differences are the experts' format (NVFP4 vs MXFP8), rope_theta 1e7 vs 5e6, MTP module, and M3.1 indexing all 60 layers (M3: 57). At 768 tokens top-16 blocks of 128 cover every key, so indexed and dense attention read the same keys. The two causes I'm fixing (selected attention 4 CTAs; single-CTA q/k norm + RoPE) are identical code paths on both cuts.
3. Cut B is slower than cut A because its experts are MXFP8 (8 bits) where A's are NVFP4: twice the expert bytes. The attribution table sums to the 2.80 ms delta (9.64 - 6.85 profiled). The MXFP8 GEMMs themselves are close to vLLM's at one row (experts gate-up 33 vs 31 us, down 18 vs 13, qkv 15 vs 13); the bigger shares are attention, q/k norm and the glue around the experts.
4. Identify bug: will write up separately with file:line and a minimal repro, not in the perf diff.
Orchestrator note 1: acknowledged. Disk after the first cut: df -h / = 95G used / 156G free of 250G; one copy of each cut, shared by all engines.
