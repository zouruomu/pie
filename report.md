# Report: B200 × MiniMax-M3.1-Flash (miniature) × decode

Draft, in progress. Anything not yet measured is marked pending.

## The cut and its parameters (note 1)

- Cut A: `MiniMaxAI/MiniMax-M3.1-Flash` @ce7f4df8 (weights unchanged since the 10-23 init), `scripts/bench/shrink_checkpoint.py --layers 0-59 --experts 16`: 43.41 GiB, 13 024 tensors, MXFP8 projections, NVFP4 experts kept as released. The site's row (`data/tags/b200-full.json`, pie 0d857cda6) states `base_model` MiniMaxAI/MiniMax-M3.1-Flash, `mini_layers` 60, `mini_k` 0.125, `mini_params_b` 124.25, `mini_active_b` 23.0, `mini_emb_b` 2.458.
- My count (`results/scripts/count_params.py`, NVFP4 planes counted two values a byte; `results/param_count.txt`): text weights 64.67 B (routed 51.64, attention/dense/router/norms 7.34, shared expert 3.23, embedding + lm_head 2.458), active 25.9 B (top-4 + shared + the rest); block scales 3.56 B and the vision tower 0.87 B apart.
- The embedding matches the site exactly. 124.25 B matches no integer expert count: with 56.6 M parameters per expert over 57 routed layers it would take about 35 of 128 experts, while `mini_k` 0.125 is 16 of 128, the cut I made. I read 124.25 / 23.0 as the site's own formula, not a different cut. For decode at concurrency 1 it can't matter: each routed layer reads top-4 + shared experts whatever the bank size.
- Cut B: `MiniMaxAI/MiniMax-M3-MXFP8` cut the same way: 65.95 GiB, same architecture (60 layers, 128 experts, top-4, hidden 6144), every expert MXFP8. Text weights 64.65 B, active 25.9 B (`results/param_count_m3mxfp8.txt`).

## Why two cuts

- vLLM 0.32.0, vLLM 0.33.0 and SGLang 0.5.22 cannot load cut A: each fails with "size of tensor a (192) must match the size of tensor b (384)" (`results/raw/vllm-r1.log`, `vllm033-r1.log`, `sglang-r1.log`). They read the NVFP4 experts' 1x16 scales (K/16 = 384) as MXFP8 1x32 scales (K/32 = 192). vLLM's recipe and SGLang's cookbook serve M3.1-Flash only from dev images (`vllm/vllm-openai:minimax-m31-flash`; `lmsysorg/sglang:dev-minimax-m31-flash`, "support not in main yet").
- So the site's vLLM 0.32.0 and SGLang 0.5.22 numbers were not measured on this checkpoint at those versions. Cut B is the same architecture in the form those versions do load ("the MXFP8 release": the site's scheme wording, and the comment on `minimax_m3_vl` in shrink_checkpoint.py), so it is the fair same-weights comparison I can reproduce.

## Side finding: pie serves both cuts as bf16 by default

- `crates/runtime/src/engine/load.rs:267` (`identify`) returns the first deployment whose contract compiles. For a miniature, `Catalog::candidates_of` tries the cut deployments in package order, and `models/minimax_m3.poem` lists `minimax-m3-mini`'s bf16 deployment first. So both cuts load as `minimax-m3-mini-bf16-kv-bf16-l60-e16`: the MXFP8/NVFP4 planes widen to bf16 (124.5 GB on the GPU) and the load needs more than this pod's 264 GiB cgroup on the host (OOM-killed twice, `results/logs/mem-r1b.log`). `--precision` can't override it: the engine refuses after identification ("stores its weights at [Bf16] and the config asks for [Mxfp8, Nvfp4]").
- For every measurement here pie serves the release precision, via the package order in `~/.pie/models/minimax_m3.poem` (the mini's [MX8, nvfp4a4] and MX8 deployments listed ahead of bf16, a local edit only). Every pie log from `pie-m3-r2` on records its deployment (the wrapper sets `RUST_LOG=runtime::engine::load=info` from there): `minimax-m3-mini-mxfp8a8g32-nvfp4a4-kv-bf16-l60-e16` on cut A (all A/B rounds), `minimax-m3-mini-mxfp8a8g32-kv-bf16-l60-e16` on cut B. The four earlier pie runs (`pie-mx8nv-probe`, `pie-m31-r2/r3`, `pie-m3-r1`) ran the same engine and package edit at the same 44 / 67 GB footprints (`results/logs/mem-*.log`) and timings.

## Reproduced baseline (B200, pie main e2e9d1e)

Raw workload: concurrency 1, 512-token chat-rendered prompts (`results/scripts/make_prompts.py`, one file a cut since the chat templates differ; pie's inferlet renders them to 497 tokens), 256 output tokens, greedy, 8 requests after 2 warmups. Decode tok/s = 1000 / the median of the rounds' exact ITL p50 (`results/scripts/numbers.py`). These runs predate the in-run clock sampler: their `.meta` holds idle start/end clocks only; the A/B and cross-model runs below carry a busy-clock median (1965 MHz every time).

| cut | engine | ITL p50, three rounds (ms) | decode tok/s | logs |
|---|---|---|---|---|
| A (M3.1-Flash) | pie, MX8 + NVFP4 deployment | 9.01 / 8.98 / 8.97 | 111.4 | `raw/pie-mx8nv-probe`, `pie-m31-r2`, `pie-m31-r3` |
| A | vLLM 0.32.0 / 0.33.0, SGLang 0.5.22 | cannot load | - | `raw/vllm-r1`, `vllm033-r1`, `sglang-r1` |
| B (M3-MXFP8) | pie, MX8 deployment | 9.62 / 9.62 / 9.62 | 103.9 | `raw/pie-m3-r1..3` |
| B | vLLM 0.32.0 | 6.92 / 6.88 / 6.87 | 145.2 | `raw/vllm-m3-r1s..r3s` |
| B | SGLang 0.5.22 | 7.59 / 7.52 / 7.52 | 132.9 | `raw/sglang-m3-r1..3` |

- Site (pie 0d857cda6): pie 8.99 ms, vLLM 5.73, SGLang 7.80. Pie on cut A reproduces the site's 8.99 to 0.2%. SGLang on cut B is within 4% of its site number; vLLM is 20% slower here than on the site (6.88 vs 5.73), consistent with the site's vLLM not running this checkpoint at 0.32.0.
- Mode: vLLM and SGLang run the bench's streaming client (`tput --concurrency 1 --report-timing`), pie its latency client; pie in the same streaming mode gives 9.66 ms on cut B (`raw/pie-m3-tput-r1`) against 9.62, so the mode doesn't move ITL. SGLang's client submits all requests at t=0, so its TTFT there is queueing; its ITL is one sequence at a time.
- Gap on cut B: pie trails vLLM by 28.5% and SGLang by 21.8%.
- Config differences between the two checkpoints (`results/analysis/config-diff-m3-vs-m31.txt`): the experts' format (NVFP4 vs MXFP8), rope_theta (1e7 vs 5e6), an MTP module in M3, and M3.1 indexing all 60 layers where M3 indexes 57. At 768 tokens the indexer's top-16 blocks of 128 cover every key, so indexed and dense layers read the same keys. Shapes are identical. Cut B is slower in pie than cut A most likely because its experts are MXFP8 (8.25 bits a value with scales against NVFP4's 4.5, 1.83x the expert bytes); not attributed separately.

## Per-step attribution (cut B, nsys `--cuda-graph-trace=node`)

`results/scripts/components.py`, `results/analysis/components-{pie,vllm}-m3.tsv`. Each kernel is charged its exposed time (its end minus the latest earlier end on any stream), so overlapped kernels (pie's PDL launches, vLLM's second stream) aren't double-counted, and what's left of the step period is "gaps". Profiled step: pie 9640 us, vLLM 6845 us (unprofiled 9.62 / 6.88 ms).

| component | pie | vLLM | delta |
|---|---|---|---|
| GEMMs: projections + routed experts | 4117 | 2649 | +1468 |
| norms, q/k norm + RoPE, KV append | 1412 | 602 | +810 |
| attention (selected + 3 dense layers) | 1092 | 409 | +683 |
| routing: gate, top-k, layout, gather, weighted sum | 1106 | 589 | +517 |
| activation quantize (e4m3 1x32) | 863 | 461 | +402 |
| SwiGLU | 271 | 190 | +81 |
| index top-k (and vLLM's index scores) | 102 | 243 | -141 |
| lm_head | 350 | 404 | -55 |
| other (sampling, bookkeeping) | 37 | 78 | -41 |
| gaps (no kernel running) | 290 | 1220 | -930 |
| **step** | **9640** | **6845** | **+2795** |

The GEMM row splits differently by overlap (vLLM runs the shared expert on a second stream beside the routed experts, so most of its routed time hides); on the critical path, layer by layer (one MoE layer of one step, pie 155 us vs vLLM ~108 us; each entry is the interval from the previous piece's end to this one's on that layer's timeline, not a kernel median: vLLM's q/k norm kernel median is 2.66 us and its attention kernels' 4.93 + 2.24 us, `results/analysis/nsys-*-kernels.txt`):

| piece of an MoE layer | pie (us) | vLLM (us) | delta |
|---|---|---|---|
| quantize + fused q/k/v(/index) projection | 19.8 | 13.9 | +5.9 |
| q/k norm + RoPE + KV append | 14.4 (two 1-CTA fused row skeletons) | 2.3 (`fusedMiniMaxM3QNormRopeKVInsertKernel`, 10 CTAs) | +12.1 |
| index top-k | 2.1 | 3.9 (score + top-k) | -1.8 |
| selected attention | 18.5 (4 CTAs) | 6.9 (16 splits a kv head + merge) | +11.6 |
| quantize + o_proj | 15.0 | 12.7 | +2.3 |
| residual + norm, gate | 8.1 | 7.3 | +0.8 |
| routing layout + quantize | 10.2 | 5.9 | +4.3 |
| experts gate-up | 33.2 | 31.0 | +2.2 |
| glue between gate-up and down (gather, SwiGLU, layout, quantize) | 11.2 | ~0 (activation in vLLM's down bmm) | +11.2 |
| experts down | 18.3 | 12.8 | +5.5 |
| weighted sum / finalize | 4.2 | 6.0 | -1.8 |

Of these, three are each worth 0.6-0.7 ms a step: the q/k norm + RoPE, the selected attention, and the expert glue. The attention and the q/k norm are the same code on cut A (selected attention 18.4-18.6 us a layer there too; the expert path differs: NVFP4 W4A4 on tcgen05).

## Root cause fixed here: the selected attention reads a kv head on one block

- pie: `crates/kernels-cuda/src/attn/selected.rs:210-227` (at e2e9d1e) launches `sdpa_paged_selected_mma_kernel` (`crates/kernels-cuda/kernels/attn/selected.cuh:43`) on a grid of (rows, kv heads): a decode's lone row of MiniMax's four kv heads is four blocks on a 148-SM GPU. Each block's warps walk 16-key chunks of the selected keys (498-753 a row over this run, ~625 on average; up to ~390 KB of K and V a kv head), so the walk is bound by what one SM can pull and by each chunk's load latency. The 16-warp widening (`mma_warps`, `selected.rs:44`) already acknowledges the lone-row case but keeps one block.
- vLLM 0.32.0: `vllm/models/minimax_m3/common/ops/sparse_attn.py:240` (`_gqa_sparse_decode_kernel`) splits the selected blocks into shape-constant chunks, grid `(total_q * num_topk_chunks, num_kv_heads)` (`sparse_attn.py:662-673`), 16 splits a kv head here, and `_merge_topk_attn_out_kernel` (`sparse_attn.py:434`) folds the partials: 5.0 + 2.2 us.

## The fix, line by line (commit `3da03d2c1` on `r2g/20261106-0130-b200-x1-minimax-m3-1-flash-decode`, 8 files, +233 / -47)

- `crates/kernels-cuda/kernels/attn/selected.cuh:42-45`: `sdpa_paged_selected_mma_kernel` gains a template parameter `kRanks` (default 1, the old kernel), with a three-line comment.
- `selected.cuh:66-67`: the block's row is `blockIdx.x / kRanks`, its rank in the cluster `blockIdx.x % kRanks` (a cluster is `kRanks` consecutive blocks along x, so all ranks of a cluster share the row and the `win` early return).
- `selected.cuh:127`: the key-chunk loop starts at `(rank * kWarps + warp) * kKeys` and strides `kRanks * kWarps * kKeys`: each block walks a slice of the chunks, its warps as before.
- `selected.cuh:268-276`: with `kRanks > 1` the block's fold over its warps lands its normalized output and its log-sum-exp (`m + ln l`, natural base as the kernel's softmax) in shared memory past the kernel's own (`S::kBytes`), instead of writing the output.
- `selected.cuh:278-307`: after a cluster barrier, each block folds its share of the (row, column) cells across the cluster's slices, reading their states through distributed shared memory (`ld_cluster`): weights `exp(lse_r - max lse)`, output `sum w_r o_r / sum w_r`; a slice with no keys has `lse = -inf` and weight 0. A second barrier keeps every block's shared memory alive until the others have read it. The same fold as `decode_gqa_sm80.cu`'s clusters.
- `crates/kernels-cuda/kernels/prim/device.cuh` + `attn/decode_gqa_sm80.cu`: `ld_cluster` and `sync_cluster` move from `decode_gqa_sm80.cu` into the shared header, byte for byte, so both kernels use one copy.
- `crates/kernels-cuda/src/attn/selected.rs:53-75`: `mma_ranks`: the profile's cluster size, halved while the fire's clustered blocks would fill more than a quarter of the SMs, on sm_90 and later where the fold's shared memory fits; else 1. On a B200 (148 SMs), where it turns on and off by rows in the fire: MiniMax (4 kv heads) 1 row 8 ranks, 2 rows 4, 3-4 rows 2, 5 rows and more off; Qwen3.8-Flash-Next (2 kv heads) 1-2 rows 8, 3-4 rows 4, 5-9 rows 2, 10 and more off. `mma_fold_bytes` sizes the fold's slice (8256 bytes at 128-wide heads).
- `selected.rs:234-263`: 8 warps a block under a cluster (the sweep's best), a grid of `rows * ranks` by kv heads, `.cluster(ranks)` and the fold's shared memory; without a cluster the warps, launch and kernel name are exactly as before.
- `crates/kernels-cuda/src/tuning.rs:487-494, 1071, 1234`: the decision `kernel.sdpa.selected_cluster` (0, 2, 4 or 8, the portable cluster sizes; default 0, so no other chip changes).
- `crates/engine-cuda/profiles/sm100/nvidia-b200.toml:14-17`: `sdpa.selected_cluster = 8` beside `sdpa.selected_mma`, with its measured win.
- `crates/engine-cuda/src/chip.rs:841`: the B200-vs-family test lists the new setting among the B200's own (within the sm100 family only the B200 device profile sets `selected_mma`; the cluster sits beside it).
- `crates/kernels-cuda/tests/kernels_cuda/attn/selected.rs`: the existing test checks the tensor-core walk against the scalar walk a third time with `selected_cluster = 8` (its rows hold at most 27 keys, so only rank 0 walks keys there); a new test, `a_clusters_slices_fold_into_what_one_block_walks` (line 195), takes MiniMax's decode shape (16 query heads a kv head of 128, a selection a kv head, 128-key blocks) over one 1100-key row with its pages out of order, so every rank of the cluster walks keys for the first kv head and two ranks walk none for the second (two empty selection slots), and checks the one-block and clustered walks against the scalar walk.

Why 8 ranks and 8 warps: a single-round sweep on cut A (4 requests each, `results/raw/sweep-a-*`; an env-var build that clustered every tensor-core fire, prefill too, which is why its TTFTs run 38-90 ms): ranks x warps 1x16 (base) 8.98, 8x4 8.77, **8x8 8.63**, 4x8 8.81, 8x16 8.80, 4x16 8.77 ms. 8x8's lead is well outside run-to-run spread (0.01 ms); the order of the rest isn't. 8 is the portable cluster limit.

Why the halving: on the first fix build (any fire under a quarter of the SMs took 8 ranks), cut A c4 (16 blocks, 128 clustered) was 9.72 -> 9.73 ms and c8 (32 blocks, 256 clustered) 10.11 -> 10.21 ms (`results/raw/curve-fix1/`): no gain at c4, a 1% loss at c8. The final rule keeps the clustered blocks within a quarter of the SMs (c4 takes 2 ranks, c8 none); its curve is below.

## Before / after (final commit, 3 rounds, `results/raw/fin-*`)

Engines (`results/engines.sha256`): base `b0ca94bb…` (main e2e9d1e), final `fff664e3…` (3da03d2c1). Each round ran base then fix on cut A, then base then fix on cut B (interleaved, not order-balanced). Every `.meta` holds the engine sha256 and a busy SM clock median of 1965 MHz.

| cut | base ITL p50 (ms) | fix ITL p50 (ms) | decode tok/s (1000 / median) | vLLM 0.32.0 ITL p50 (ms) |
|---|---|---|---|---|
| A, M3.1-Flash (the site's checkpoint) | 8.976 / 8.974 / 8.978 | **8.620 / 8.620 / 8.618** | 111.4 -> 116.0 (+4.1%) | can't load |
| B, M3-MXFP8 | 9.623 / 9.621 / 9.627 | **9.286 / 9.284 / 9.299** | 103.9 -> 107.7 (+3.6%) | 6.925 / 6.868 / 6.969 (`ab-b-vllm-r*`, same session) |

- Cut B against vLLM's same-session rounds (median 6.925 ms, 144.4 tok/s): pie behind by 28.0% before, 25.4% after. Against the site's vLLM number (174.43) cut A goes from -36.1% to -33.5%, but that number isn't this checkpoint at 0.32.0 (see above).
- The first fix build (`2245691c…`, commit 98dccdc21, before the review round) measured the same at c1 (`results/raw/ab-*`: A 8.62 x3, B 9.28 / 9.29 / 9.30) with identical tokens to the final build on both cuts: at one row both take 8 ranks, so the c1 decode path is the same kernel.
- nsys (cut A, the sweep build at the same 8 ranks x 8 warps, trace `sel-r8w8-node` on the pod): `sdpa_paged_selected_mma_kernel<128, 8, 8>` on a grid of 8 x 4 blocks in clusters of 8, kernel median 11.71 us against the base's `<128, 16>` on 1 x 4 blocks, 18.37 us; on the critical path 18.4 -> 11.8 us a layer.
- TTFT p50 (pie, c1): cut A base 28.4 / 28.0 / 27.9 ms (`ab-a-base-*`), fix 27.8 / 28.2 / 28.2 (`ab-a-fix-*`); cut B 31.5 -> 31.7 (round 1). This workload's ~500-token prefill doesn't take the cluster (a fire of more than 9 rows at 4 kv heads doesn't); a prefill of a few rows would.

## Tokens and numerics

- Greedy token ids are deterministic run to run in both builds (r1 vs r3: 8/8 identical, both cuts).
- Cut B: base vs fix **identical**, 8/8 requests x 256 tokens. At ~500-750 keys a row ranks 0-5 of the cluster hold keys there, so the real fold runs end to end.
- Cut A: token 0 identical on every request (prefill untouched); first divergence at token 1 on 5/8, at 8 and 247 on two more, one identical.
- Teacher-forced logits (note 3): a throwaway inferlet (`results/scripts/logit-probe/`, client `logit_probe.py`, comparison `logit_cmp.py`) feeds each request's prompt as text-completion-bench renders it, then 32 tokens of base's own greedy output one decode step at a time (the clustered path), and returns all 200 064 logits at each of the 33 positions; 4 requests a cut (requests 0-3). Base's teacher-forced argmax reproduces its own greedy tokens at 33/33 positions on every request. Taken on the first fix build, whose tokens the final build matches. Reference for a reduction-order change pie already accepts: base under `sdpa.selected_mma = false` (the scalar walk: the fallback wherever a profile doesn't set `selected_mma`, and the reference `attn::selected`'s test holds the MMA walk to), set in a user profile layer and removed after. The scalar walk also changes prefill (position 0), so the comparison below is over the decode positions 1-32, with position 0 apart.

| cut, request | fix: pos 0 KL / decode max abs dlogit / KL mean / KL max / tf top-1 of 33 | scalar walk: same |
|---|---|---|
| A 0 | 0 / 4.34 / 0.025 / 0.172 / 33 | 0.30 / 5.39 / 0.087 / 1.005 / 33 |
| A 1 | 0 / 4.22 / 0.037 / 0.406 / 32 | 0.76 / 4.62 / 0.046 / 0.474 / 32 |
| A 2 | 0 / 5.26 / 0.058 / 0.741 / 31 | 0.39 / 5.67 / 0.069 / 0.844 / 31 |
| A 3 | 0 / 6.16 / 0.154 / 1.092 / 25 | 0.98 / 6.94 / 0.230 / 1.416 / 24 |
| B 0 | 0 / 2.16 / 0.0038 / 0.044 / 33 | 0.07 / 2.84 / 0.0136 / 0.182 / 33 |
| B 1 | 0 / 3.19 / 0.0144 / 0.297 / 33 | 0.05 / 3.88 / 0.0161 / 0.194 / 33 |
| B 2 | 0 / 2.52 / 0.0141 / 0.138 / 33 | 0.14 / 3.21 / 0.0171 / 0.182 / 33 |
| B 3 | 0 / 4.69 / 0.0064 / 0.061 / 33 | 0.10 / 3.42 / 0.0164 / 0.164 / 32 |

- The fix leaves position 0 bit-identical; over the decode positions its mean KL to base is below the scalar walk's on all 8 probed request-cut pairs (closest: B 1, 0.0144 vs 0.0161). Its worst cases are not uniformly smaller (B 1's KL max 0.297 vs 0.194, B 3's max abs dlogit 4.69 vs 3.42).
- At the three cut A greedy divergences the probe covers, base's top-2 logit margin is 0.0625 (request 2, position 1), 0.6875 (request 3, position 1) and 1.0 (request 1, position 8), against per-request medians of 0.875-6.4 (request 3's own median is 0.875, so its 0.6875 isn't far below it). Requests 4-6 also diverge at token 1 and weren't probed. The logits land in bf16 (margins come in steps of 1/16), and the scalar walk flips the same two position-1 tokens at the same margins.
- Several logit units of movement under either order are this miniature's sensitivity (16 of 128 experts with top-4: a small change in a layer's input can swap one of its experts), not the kernel's error: at the kernel level the cluster walk is within the existing test's bf16 tolerance of the scalar walk, on a row whose slices all hold keys (the new test).

## Cross-model (B200, base vs fix, 2 rounds each, raw workload, `results/raw/x-*`)

| model | path | base ITL p50 (ms) | fix ITL p50 (ms) | tokens base vs fix |
|---|---|---|---|---|
| Qwen3.8-Flash-Next mini (4 layers, 16 experts, bf16; cut with shrink_checkpoint `--experts 16`) | takes the selected kernel (2 kv heads of 256, a shared selection): cluster on at decode | 0.85 / 0.85 | 0.82 / 0.82 (+3.7%) | diverge at 38-229 on 7/8, one identical; deterministic within each build |
| Qwen3-0.6B bf16 | dense tensor-core decode with clusters (`gqa_cluster = 8`), whose two helpers moved to `prim/device.cuh` | 1.03 / 1.03 | 1.03 / 1.03 | identical 8/8 |

- Only GQA models with selected attention take the changed kernel: MiniMax-M3 and Qwen3.8-Flash-Next (`qwen_4.poem`). The MLA models with selections (GLM-5.x, DeepSeek-V4, Kimi-K3) go through `attn/mla` kernels, untouched. Every other model sees only the helper move, a byte-for-byte move of two device functions, which Qwen3-0.6B's identical tokens and speed confirm on the cluster path that uses them. So no block-FP8 MoE beyond the two cuts was rerun; GLM-5.3's cut (MLA) doesn't reach the change.
- Memory: the fix adds 8256 bytes of shared memory to a clustered block (`mma_fold_bytes`) and no device or host allocation; nothing starts or stops fitting.

## Other loads (pending)

Base vs fix on both cuts, one round, `results/raw/curve/`: batching c4, c8, c16, c32, c64 (256-token prompts, 128 out; the chat template alone renders past 128 tokens), prefill batch pf4-pf32 (512 in, 1 out). The cluster applies where a fire's (row, kv head) blocks are fewer than a quarter of the SMs: at 4 kv heads, 9 rows or fewer, so c4 and c8 take it, c16 and up and every prefill don't. PENDING.

## Tests (pending)

- `kernels-cuda` `attn::selected` (extended with the cluster walk) and `attn::fa2*`: pass on the fix build (`results/tests/kernels-selected-fa2-fix1b.log`); a deliberate error in the cluster fold fails `attn::selected` at walk 2, so the test reaches the new path.
- `engine-cuda` unit tests: 68/69 on the first fix build, the failure being `chip::tests::the_b200_runs_its_family_measured_with_the_skinny_gemm_off`, which lists the B200's own settings and now lists `selected_cluster` (`results/tests/engine-cuda-lib-fix1b.log`, then `engine-cuda-lib-chip-fix1c.log`: 12/12 chip tests).
- GPU suite (engine-cuda and kernels-cuda, `--test-threads=1`, `CARGO_PROFILE_DEV_OPT_LEVEL=2`) on the final diff: PENDING.

## Fix 2, measured but not shipped: the q/k norm + RoPE + KV append

- Mechanism: MiniMax's q/k per-head norms (plus-one weights), partial RoPE and KV append are folded as an anchored epilogue of the packed q|k|v MXFP8 product (`Fused::Epilogue`). The W8A8 tcgen05 GEMM doesn't run it in-kernel (`crates/engine-cuda/src/dispatch/linear.rs:350`, `anchored`, takes dense bf16 banks), so it fires detached (`dispatch/linear.rs:393` -> `crates/kernels-cuda/src/linear/gemm.rs:136` -> `RowLeg::fire`, `linear/skeleton.rs:1259`). `Epilogue::leg` (`skeleton.rs:718`) cuts the 9216-wide row into 2 slices of 4608 (each no wider than the skeleton holds); the results are windowed and paged, so the slices aren't uniform and each is its own launch of one block a row: two 1-CTA kernels, ~12.7 us a layer on the critical path on cut A (14.4 us on cut B's per-layer timeline). vLLM: one `fusedMiniMaxM3QNormRopeKVInsertKernel` of 10 CTAs, kernel median 2.66 us (`vllm/models/minimax_m3/nvidia/model.py:625`, `vllm/_custom_ops.py:2566`).
- The hand kernel for this shape exists (`fused.qkv_fused_qknorm_rope_vnorm_write`, `crates/kernels-cuda/kernels/attn/qkv_append.cu`; its launcher takes MiniMax's span 32 / rotary 64 at 128-wide heads) but lacks plus-one weights, and CUDA retired that rule on purpose (`crates/engine-cuda/src/lib.rs:106-107`: "every site one forms folds into a generated row kernel or an anchored region instead"; the retired list at 110-127).
- Prototype, to size it (reverted, never committed): the retired rule plus a plus-one MiniMax pattern registered on CUDA, plus-one in the warp and row kernels, on top of the first fix build. One round of 4 requests each (`results/raw/proto2-*`): cut A 8.62 -> **8.27 ms** (-4.0%, busy clock 1965 MHz), cut B 9.29 -> **8.88 ms** (-4.4%, but that run's busy clock median was 1792 MHz, so the cut B number is indicative only). Logits (`results/analysis/logit-cmp.txt`, "base vs proto2"): position 0 moves (KL 0.6-1.3 on A, 0.06-0.21 on B: prefill takes the hand kernel too), and over the decode positions its mean KL to base is 0.073-0.178 on cut A (scalar walk 0.046-0.230) and 0.015-0.039 on cut B (scalar walk 0.014-0.017, so 0.9x to 2.4x the scalar's, request by request); greedy tokens identical to base on cut B (4/4), different from token 0-2 on cut A. The hand kernel rounds once where the generated region rounds at the program's points.
- In the maintainers' direction the fix is one of: (a) the anchored epilogue inside the block-scaled tcgen05 GEMM; (b) a detached leg that spreads a row's non-uniform slices over many blocks in one launch (the slice offset a runtime value, not baked into each piece's kernel). I'd put either past this run's verification budget; the prototype puts the win at ~4%.

## Remaining gap (cut B, after the fix: pie 9.29 ms vs vLLM 6.92 ms, +2.37 ms)

From the per-layer table with the attention's measured gain taken out (us a layer x 57, approximate; vLLM's profiled gaps of ~1.2 ms a step don't all appear unprofiled, which is why the pieces sum past 2.37):

| piece | delta a layer (us) | a step (ms) |
|---|---|---|
| q/k norm + RoPE + KV append (two 1-CTA epilogue slices) | +12.1 | +0.69 |
| expert glue between gate-up and down (MXFP8 experts: gather, SwiGLU, layout, quantize as separate launches; vLLM's down bmm takes the activation) | +11.2 | +0.64 |
| selected attention, still (11.8 vs 6.9 us; my unmeasured guess is the chain of dependent metadata loads before the K/V reads) | +4.9 | +0.28 |
| qkv projection (quantize + tcgen05 GEMM at one row) | +5.9 | +0.34 |
| experts down GEMM | +5.5 | +0.31 |
| routing layout + quantize | +4.3 | +0.25 |
| o_proj, experts gate-up, norms | +5.3 | +0.30 |
| index, finalize | -3.6 | -0.21 |

On cut A (the site's checkpoint, NVFP4 experts) the expert path differs: the W4A4 gate-up runs at ~2.3 TB/s (33 us for ~75 MB; swapped plan, `splits: 1`, 192 tiles on 148 SMs, `crates/kernels-cuda/src/linear/moe/experts.rs:566-572`) and `route_lay_out_quant_nvfp4` (6 CTAs) costs ~10 us a layer exposed. Each is kernel work past ~3 h (note 4's bar), so it's handed over as attribution.

The cause is general rather than sm_100-specific in part: the selected walk reads a kv head on one block on every GPU that sets `selected_mma` (the H200 and A100 cells are red too), and clusters exist from sm_90; only B200's profile turns the cluster on here, since it's the only GPU I measured.

## Side findings

1. **No released vLLM or SGLang loads MiniMax-M3.1-Flash** (`results/raw/vllm-r1.log`, `vllm033-r1.log`, `sglang-r1.log`); the site's vLLM 0.32.0 / SGLang 0.5.22 numbers on this cell can't be this checkpoint at those versions.
2. **Pie identifies both cuts as bf16** and OOMs widening them on a 264 GiB host (`identify-bug.md`, with `file:line` and a repro; kept out of this PR per note 2).
3. **Pie reads M3.1-Flash's first three layers as dense attention** although its config indexes all 60 (`sparse_attention_freq` sums to 60; layers 0-2 carry indexer weights the poem doesn't read). Identical at this length (top-16 x 128 >= every key), different past 2048 tokens.
4. The site's 124.25 B parameter count doesn't follow from a 16-expert cut (64.67 B; see the first section).
5. `make_prompts.py` can't build the site's 128-token batching prompts for MiniMax: its chat template with the bench's system prompt alone renders past 200 tokens.

## Playbook notes

- When a site cell names a competitor version, check that version can load the checkpoint before profiling against the number: here neither released competitor could.
- `pgrep -f` inside a wait loop matches the loop's own shell: wait on log contents or PIDs.
- zsh doesn't word-split `$var` (`set -- $cfg` silently runs one config): use `${=cfg}`.
- A seeded package under `~/.pie/models/` can be edited to steer identification without a rebuild; pie keeps the edit (and reports it as edited). Restore it after.
- An inferlet can return full logits through `session::send` in pieces of 32 K values (a ~1 MB message closes the socket); `results/scripts/logit-probe` is a reusable teacher-forced logit probe.
