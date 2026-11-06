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
- For every measurement here pie serves the release precision, via the package order in `~/.pie/models/minimax_m3.poem` (the mini's [MX8, nvfp4a4] and MX8 deployments listed ahead of bf16, a local edit only). Each pie log records its deployment: `minimax-m3-mini-mxfp8a8g32-nvfp4a4-kv-bf16-l60-e16` (cut A), `minimax-m3-mini-mxfp8a8g32-kv-bf16-l60-e16` (cut B).

## Reproduced baseline (B200, pie main e2e9d1e, SM clock 1965 MHz in-run)

Raw workload: concurrency 1, 512-token chat-rendered prompts (`results/scripts/make_prompts.py`, one file a cut since the chat templates differ), 256 output tokens, greedy, 8 requests after 2 warmups. Decode tok/s = 1000 / ITL p50.

| cut | engine | ITL p50, three rounds (ms) | decode tok/s | logs |
|---|---|---|---|---|
| A (M3.1-Flash) | pie, MX8 + NVFP4 deployment | 9.01 / 8.98 / 8.97 | 111.4 | `raw/pie-mx8nv-probe`, `pie-m31-r2`, `pie-m31-r3` |
| A | vLLM 0.32.0 / 0.33.0, SGLang 0.5.22 | cannot load | - | `raw/vllm-r1`, `vllm033-r1`, `sglang-r1` |
| B (M3-MXFP8) | pie, MX8 deployment | 9.62 / 9.62 / 9.62 | 104.0 | `raw/pie-m3-r1..3` |
| B | vLLM 0.32.0 | 6.92 / 6.88 / 6.87 | 145.4 | `raw/vllm-m3-r1s..r3s` |
| B | SGLang 0.5.22 | 7.59 / 7.52 / 7.52 | 133.0 | `raw/sglang-m3-r1..3` |

- Site (pie 0d857cda6): pie 8.99 ms, vLLM 5.73, SGLang 7.80. Pie on cut A reproduces the site's 8.99 to 0.2%. SGLang on cut B is within 4% of its site number; vLLM is 20% slower here than on the site (6.88 vs 5.73), consistent with the site's vLLM not running this checkpoint at 0.32.0.
- Mode: vLLM and SGLang run the bench's streaming client (`tput --concurrency 1 --report-timing`), pie its latency client; pie in the same streaming mode gives 9.66 ms on cut B (`raw/pie-m3-tput-r1`) against 9.62, so the mode doesn't move ITL. SGLang's client submits all requests at t=0, so its TTFT there is queueing; its ITL is one sequence at a time.
- Gap on cut B: pie trails vLLM by 28.5% and SGLang by 21.8%.
- Config differences between the two checkpoints (`results/analysis/config-diff-m3-vs-m31.txt`): the experts' format (NVFP4 vs MXFP8), rope_theta (1e7 vs 5e6), an MTP module in M3, and M3.1 indexing all 60 layers where M3 indexes 57. At 768 tokens the indexer's top-16 blocks of 128 cover every key, so indexed and dense layers read the same keys. Shapes are identical. Cut B is slower in pie than cut A because its experts are MXFP8: twice the expert bytes.

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

The GEMM row splits differently by overlap (vLLM runs the shared expert on a second stream beside the routed experts, so most of its routed time hides); on the critical path, layer by layer (one MoE layer, pie 155 us vs vLLM ~108 us):

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

- pie: `crates/kernels-cuda/src/attn/selected.rs:210-227` (at e2e9d1e) launches `sdpa_paged_selected_mma_kernel` (`crates/kernels-cuda/kernels/attn/selected.cuh:43`) on a grid of (rows, kv heads): a decode's lone row of MiniMax's four kv heads is four blocks on a 148-SM GPU. Each block's warps walk 16-key chunks of the ~770 selected keys (a kv head's ~390 KB of K and V), so the walk is bound by what one SM can pull and by each chunk's load latency. The 16-warp widening (`mma_warps`, `selected.rs:44`) already acknowledges the lone-row case but keeps one block.
- vLLM 0.32.0: `vllm/models/minimax_m3/common/ops/sparse_attn.py:240` (`_gqa_sparse_decode_kernel`) splits the selected blocks into shape-constant chunks, grid `(total_q * num_topk_chunks, num_kv_heads)` (`sparse_attn.py:662-673`), 16 splits a kv head here, and `_merge_topk_attn_out_kernel` (`sparse_attn.py:434`) folds the partials: 5.0 + 2.2 us.

## The fix, line by line (commit `98dccdc21` on `r2g/20261106-0130-b200-x1-minimax-m3-1-flash-decode`, 8 files, +138 / -51)

- `crates/kernels-cuda/kernels/attn/selected.cuh:42-45`: `sdpa_paged_selected_mma_kernel` gains a template parameter `kRanks` (default 1, the old kernel).
- `selected.cuh:66-67`: the block's row is `blockIdx.x / kRanks`, its rank in the cluster `blockIdx.x % kRanks` (a cluster is `kRanks` consecutive blocks along x).
- `selected.cuh:127`: the key-chunk loop starts at `(rank * kWarps + warp) * kKeys` and strides `kRanks * kWarps * kKeys`: each block walks a slice of the chunks, its warps as before.
- `selected.cuh:268-276`: with `kRanks > 1` the block's fold over its warps lands its normalized output and its log-sum-exp (`m + ln l`, natural base as the kernel's softmax) in shared memory past the kernel's own (`S::kBytes`), instead of writing the output.
- `selected.cuh:278-307`: after a cluster barrier, each block folds its share of the (row, column) cells across the cluster's slices, reading their states through distributed shared memory (`ld_cluster`): weights `exp(lse_r - max lse)`, output `sum w_r o_r / sum w_r`; a second barrier keeps every block's shared memory alive until the others have read it. The same fold as `decode_gqa_sm80.cu`'s clusters.
- `crates/kernels-cuda/kernels/prim/device.cuh` + `attn/decode_gqa_sm80.cu`: `ld_cluster` and `sync_cluster` move from `decode_gqa_sm80.cu` into the shared header, unchanged, so both kernels use one copy.
- `crates/kernels-cuda/src/attn/selected.rs:53-73`: `mma_ranks` takes the profile's cluster size where the fire's (row, kv head) blocks are too few to fill a quarter of the SMs (the condition under which `mma_warps` already widens to 16 warps) and the shared memory fits; else 1. `mma_fold_bytes` sizes the fold's slice.
- `selected.rs:232-253`: the launch takes 8 warps a block under a cluster (the sweep's best), a grid of `rows * ranks` by kv heads, `.cluster(ranks)` and the fold's shared memory; without a cluster the launch and the kernel's name are exactly as before.
- `crates/kernels-cuda/src/tuning.rs:487-493, 1070, 1233`: the decision `kernel.sdpa.selected_cluster` (0, 2, 4 or 8; default 0, so no other chip changes).
- `crates/engine-cuda/profiles/sm100/nvidia-b200.toml:14-17`: `sdpa.selected_cluster = 8` beside `sdpa.selected_mma`, with its measured win.
- `crates/engine-cuda/src/chip.rs:841`: the B200-vs-family test lists the new setting among the B200's own (it sits with `selected_mma`, which only the B200 profile sets).
- `crates/kernels-cuda/tests/kernels_cuda/attn/selected.rs`: the tensor-core walk is checked against the scalar walk a third time with `selected_cluster = 8`; the lone-row decode and prefill cases take the cluster.

Why 8 ranks and 8 warps: sweep on cut A (4 requests each, `results/raw/sweep-a-*`): ranks x warps 1x16 (base) 8.98, 8x4 8.77, **8x8 8.63**, 4x8 8.81, 8x16 8.80, 4x16 8.77 ms. 8 is the portable cluster limit.
