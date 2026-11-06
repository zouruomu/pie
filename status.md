# Status: 20261106-0130-b200-x1-minimax-m3-1-flash-decode

UTC: 2026-11-06 05:25Z
Phase: final verification of fix 1 (the final diff; 98dccdc21 pushed to origin). Running: base vs fix curves both cuts (~06:30Z), then the GPU suite (~1.5 h), then self-review and final rebase A/B. report.md and pr.md drafted (pending: curves, suite). ETA hand-over ~09:30Z.

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

## A/B on the committed fix (3 alternating rounds, results/raw/ab-*, engine sha256 in each .meta: base b0ca94bb, fix 2245691c; SM 1965 MHz in-run)
| cut | base ITL p50 ms | fix ITL p50 ms | decode tok/s | vLLM 0.32.0 |
|---|---|---|---|---|
| A (M3.1-Flash) | 8.97 / 8.97 / 8.97 | 8.62 / 8.62 / 8.62 | 111.5 -> 116.0 (+4.1%) | can't load |
| B (M3-MXFP8) | 9.62 / 9.63 / 9.63 | 9.28 / 9.29 / 9.30 | 103.8 -> 107.6 (+3.6%) | 6.92 / 6.87 / 6.97 |
Tokens: deterministic run to run on both builds. Cut B base vs fix identical, 8/8 requests x 256 tokens. Cut A first diverges at token 1 on 5/8 (247, 8 on two, one identical): near-tie logits on the NVFP4 miniature; token 0 (prefill) always identical.
Remaining on cut B after the fix: 9.29 vs 6.92 ms (pie 25.5% behind vLLM).

## Note 3 (04:02Z), acknowledged
1. Numerics, done (results/analysis/logit-cmp.txt; inferlet results/scripts/logit-probe, client logit_probe.py, comparison logit_cmp.py): teacher-forced on base's own greedy tokens (prompt + 32 forced decode steps = 33 positions, 4 requests a cut, full 200064-wide logits; base's teacher-forced argmax reproduces its greedy tokens 33/33 on every request). Reference: base with `sdpa.selected_mma = false` (the scalar walk: the fallback where a profile doesn't set `selected_mma`, and the reference the unit test holds the MMA walk to), via a user profile layer, removed after.
| cut, req | fix: max abs dlogit / KL mean / tf top-1 | scalar walk: max abs dlogit / KL mean / tf top-1 |
|---|---|---|
| A 0 | 4.34 / 0.024 / 33 | 5.39 / 0.093 / 33 |
| A 1 | 4.22 / 0.036 / 32 | 4.62 / 0.068 / 32 |
| A 2 | 5.26 / 0.056 / 31 | 5.67 / 0.078 / 31 |
| A 3 | 6.16 / 0.149 / 25 | 6.94 / 0.253 / 24 |
| B 0 | 2.16 / 0.0037 / 33 | 2.88 / 0.0154 / 33 |
| B 1 | 3.19 / 0.0140 / 33 | 3.88 / 0.0170 / 33 |
| B 2 | 2.52 / 0.0137 / 33 | 3.44 / 0.0210 / 33 |
| B 3 | 4.69 / 0.0063 / 33 | 3.75 / 0.0189 / 32 |
The fix's KL is below the scalar walk's on all 8. At cut A's greedy first divergences base's top-2 margin is 0.0625 (req 2, pos 1), 0.6875 (req 3, pos 1), 1.0 (req 1, pos 8) against medians of 0.9-6.4; logits land in bf16 (margins come in 1/16 steps), and the scalar walk flips the same two positions 1 with the same margins. The miniature's logits move by several units under either reduction order (a 16-expert cut: a small change in a layer's input can change its top-4 experts).
2. Running (curves on both cuts; nsys at c32 next).
3. Fix 2 will be a separate commit.

## Note 4 (05:02Z), acknowledged: fix 2 findings so far
- Mechanism (from a temporary dump of the folded load trace): MiniMax's q/k per-head norms, partial RoPE and KV append are already folded as an anchored *epilogue* of the packed q|k|v MXFP8 product (`Fused::Epilogue`, key 6392060777819382797). The W8A8 tcgen05 GEMM can't run it in-kernel (`linear.rs:349` `anchored` takes dense bf16 banks only), so it fires detached (`dispatch/linear.rs:394` -> `kernels-cuda/src/linear/gemm.rs:136` -> `RowLeg::fire`, `linear/skeleton.rs:1263`). `Epilogue::leg` (`skeleton.rs:718`) cuts the 9216-wide row into 2 slices of 4608 (each no wider than the skeleton holds); its results are windowed and paged, so the slices are not uniform and each is its own launch of one block a row: two 1-CTA kernels, ~12.7 us a layer on the critical path.
- The hand-written kernel for this exact shape exists (`qkv_fused_qknorm_rope_vnorm_write`, `attn/qkv_append.cu`; its launcher takes MiniMax's span 32 / rotary 64 at 128-wide heads), but CUDA *retired* that rule on purpose (`engine-cuda/src/lib.rs:110-127`: "every site one forms folds into a generated row kernel or an anchored region instead"), and it lacks plus-one weights.
- So the fixes in the maintainers' direction are (a) the anchored epilogue run inside the block-scaled tcgen05 GEMM, or (b) a detached leg that spreads a row's non-uniform slices over many blocks in one launch (the slice offset a runtime value rather than baked into each piece's kernel). Both are skeleton/compiler work I'd put at more than this run's remaining verification budget.
- Prototype result (reverted, never committed; results/raw/proto2-*, analysis/logit-cmp-proto2.txt): the retired rule + a plus-one MiniMax pattern on CUDA, on top of fix 1: cut A 8.62 -> 8.27 ms (-4.0%), cut B 9.29 -> 8.88 ms (-4.4%). Its logits drift further than fix 1's: cut B KL mean 0.018-0.040, twice the scalar walk's 0.015-0.021 (cut A 0.10-0.19, inside the scalar's 0.07-0.25); prefill changes too. So fix 2 is worth ~4%, but the shippable form is (a) or (b) above, more than this run's budget: handed over as attribution in report.md, not shipped. Say if you want the hand rule anyway.
- Expert glue (note 4.2): on cut A (the site's NVFP4 path) the glue is `route_lay_out_quant_nvfp4` (6 CTAs, ~10 us a layer exposed) and the W4A4 gate-up GEMM itself runs at ~2.3 TB/s (33 us for ~75 MB: swapped plan, splits 1, 192 tiles on 148 SMs, experts.rs:566-572); on cut B the MXFP8 experts have no fused decode path (gather, SwiGLU, layout, quantize as 4 launches, +11 us a layer). Each is kernel work past ~3 h: handed over as attribution. So the final diff is fix 1 alone; the cross-model checks above were already on it.

## Cross-model (B200, base vs fix, 2 rounds each, results/raw/x-*)
- Qwen3.8-Flash-Next mini (4 layers, 16 experts, bf16; takes the selected kernel, 2 kv heads of 256): ITL 0.85 / 0.85 -> 0.82 / 0.82 ms (+3.7%); tokens deterministic per build, base vs fix diverge at 38-229 on 7/8 (one identical).
- Qwen3-0.6B bf16 (the dense tensor-core decode's clusters, whose two helpers moved to prim/device.cuh): 1.03 / 1.03 -> 1.03 / 1.03 ms, tokens identical 8/8.

## Fix 1 (details): selected attention over a cluster of 8 blocks a kv head
- `crates/kernels-cuda/kernels/attn/selected.cuh`: `sdpa_paged_selected_mma_kernel` gains `kRanks`: a cluster of that many blocks reads a (row, kv head), each a slice of the 16-key chunks, folding the slices' normalized outputs by their log-sum-exps through distributed shared memory (the fold `decode_gqa_sm80.cu` already uses). `selected.rs`: where the fire's (row, kv head) blocks are too few to fill a quarter of the SMs (where it already widened to 16 warps), 8 warps a block over `kernel.sdpa.selected_cluster` ranks; `tuning.rs` the decision (default 0); B200 profile 8.
- Sweep on cut A (ranks x warps, 4 requests): 1x16 8.98, 8x4 8.77, **8x8 8.63**, 4x8 8.81, 8x16 8.80, 4x16 8.77 ms.
- Cut A, 8 requests: ITL p50 8.98 -> **8.62 ms** (111.4 -> 116.0 tok/s, +4.2%). nsys: the attention's critical path 18.4 -> 11.8 us a layer.
- Numerics: token 0 identical on every request (prefill untouched); later tokens diverge from token 1-8 on 6/8 requests (a reduction-order change on a near-tie miniature; the bf16-GEMV experiment flipped token 1 the same way, outputs stay coherent). Kernel test `attn::selected` extended with the cluster walk against the scalar walk: passes (results/tests/kernels-selected-fix1.log); a deliberate fold error fails it at the cluster walk, so the test reaches the new path.

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
