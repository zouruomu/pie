# RTX PRO 6000 Blackwell × gpt-oss-120b / gpt-oss-20b × prefill batch: pie vs TensorRT-LLM

Run 20261106-0910-pro6000-x1-gpt-oss-120b-pbatch. pie main dc6282d76; TensorRT-LLM 1.4.0rc2.post1; vLLM 0.32.0.
Card: NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 96 GB, 600 W cap), driver 595.91.07, CUDA 13.4.

## Summary

- **Both cells are red on main, and the whole gap is the MoE's precision.** At their best points, 20b runs pie 47.0k vs TRT-LLM 72.2k tok/s (−35%) and 120b runs 29.1k vs 43.4k (−33%). The gap is the same −33…−35% from pf32 to pf1024 on both models. At pf4/pf8, pie leads or ties on 120b and trails by 7–9% on 20b.
- **TRT-LLM runs gpt-oss's experts at W4A8 on sm_120:** MXFP4 weights × MXFP8 (e4m3, one ue8m0 exponent a 32) activations, on a CUTLASS SM120 block-scaled grouped GEMM.
  - This is its only gpt-oss MoE path on sm_120. Forcing W4A16 fails with "TrtllmCutlassW4a16Mxfp4Impl supports SM90 only".
  - pie runs W4A16, which is what the release stores: bf16 activations on its register-dequant bf16 `mma.sync` selects.
- **At equal token counts, 97% of the gap is the MoE.** pf256, nsys:
  - 120b: +1,492 of +1,542 ms of kernel time is the MoE. The expert GEMMs alone are +1,660 ms, about 2× per token. pie's MoE glue is 168 ms cheaper, because its GLU and bias fuse into the select.
  - Everything else (attention, dense projections, norms, rope/KV) totals +49 ms, about 1%.
- **At W4A16 pie is level with the engines that run W4A16.** vLLM 0.32 (MARLIN W4A16) 120b pf256 is 28.9k vs pie's 28.9k. Full sweep: `pending`.
- **The 120b-specific problem in the brief does not exist on main, and did not at the site's commit either.** On this pod, ee9098f60 runs 28.7k tok/s at 120b pf256, with the same kernels and per-component times as main. The site's 13.8k for that commit was that site run. #1430 changes nothing for gpt-oss: both commits run the same W4A16 staged selects.
- **No fix PR:** per the orchestrator (note 3), changing gpt-oss's default MoE precision is out of scope. Nothing at W4A16 reaches the 4.5% bar against TRT-LLM (`pending`: vLLM sweep, the 20b pf16 dip, ea800efeb). What a W4A8 gpt-oss path on sm_120 would take, and what it would gain, is below as a decision for Felix.

## Reproduced baseline (main dc6282d76)

pending (table from results/baseline via scripts/pbtable.py)

## Fairness of TensorRT-LLM's numbers, and the precision each engine ran

The site's TRT-LLM harness is not public: pie-project/benchmarks is private, and anton-mel/stats has no TRT-LLM. So I measured it with
`scripts/trtllm_pbatch.py`, an LLM-API harness that mirrors `pie_bench.py tput`:
- the same prompts file, with 16 warmup prompts first and the same system prompt, rendered by HF's chat template to exactly 512 tokens;
- every request offered at t=0 (concurrency = N);
- `max_tokens=1`, greedy;
- prefill tok/s = prompt tokens / wall;
- all other settings at TRT-LLM's defaults, except **KV block reuse off** (cold prompts, as pie and vLLM run them; TRT-LLM's default is on).

- **Reproduces the site:** 20b 71.5k (site 71.3k), 120b 43.3k (site 42.4k), with reuse off. So the site's TRT-LLM lead is not prefix reuse.
  With reuse on, the 80 harmony tokens the prompts share (2 blocks of 32) would be skipped.
- **Same prompts, same output, best point:** 512-token prompts, 1 output token. TRT-LLM's best point is pf512–pf1024 (72.2k / 43.4k).
- **Clocks:** TRT-LLM runs at the 600 W cap at its large points, with SM clocks at 2265–2317 MHz (`clocks_before/after` per point in its JSON).
  pie draws less at the same points. `.clocks` files are sampled before and after each pie run.
- **Tokens actually prefilled:** pie 448 a prompt, TRT-LLM and vLLM 512.
  - pie's harmony template (`crates/chat-template/src/harmony.rs:142-147`) sends the system prompt as a developer message. It has no system message: no identity, knowledge cutoff, date, reasoning effort or valid-channels lines, which is 64 tokens.
  - Each engine's tok/s is over its own count. At equal counts (pie fed `prompts-gptoss-576.json`, which pie renders to exactly 512 tokens), pie runs the same per-token rate: 120b pf256 28.5k (vs 28.9k at 448), 20b 46.1k (vs 46.7k). Results in results/equal-tokens and results/prof/*-t512.

**Precision each engine ran:**

| engine | MoE experts | activations into the experts | attention / dense |
|---|---|---|---|
| pie dc6282d7 | MXFP4 (as released) | bf16 (W4A16), register-dequant bf16 `mma.sync` (`moe_select_mxfp4_mma_staged`) | bf16 |
| vLLM 0.32.0 | MXFP4 | bf16 (W4A16), MARLIN (`mxfp4.py:709` "Using 'MARLIN' Mxfp4 MoE backend") | bf16, FlashInfer |
| TRT-LLM 1.4.0rc2.post1 | MXFP4 | **MXFP8**: e4m3, a ue8m0 exponent a 32 (W4A8) | bf16 (qkv, o, router, embeddings excluded from quantization) |

TRT-LLM source, relative to site-packages/tensorrt_llm (full notes: results/fairness/trtllm-moe-precision-source.md):
- `_torch/model_config.py:579-587`: gpt-oss MoE backend AUTO → CUTLASS on sm_120.
- `_torch/model_config.py:729-741`: sm ≥ 100 and backend != TRITON → `QuantAlgo.W4A8_MXFP4_MXFP8`. W4A16 is chosen only below sm100.
- `_torch/moe/fused_moe/cutlass/w4a8_mxfp4_mxfp8.py:36-41`: `quantize_input = mxfp8_quantize`, sm {100, 103, 107, 120, 121}.
- `_torch/moe/fused_moe/cutlass/base.py:469-488`: the activations quantized every forward; `:371` `torch.ops.trtllm.fused_moe`.
- `libs/libth_common.so`: `tma_warp_specialized_generic_moe_gemm_kernelLauncher<Sm120, __nv_fp8_e4m3, __nv_fp4_e2m1, …, 128_128_128_1_1_1>`.
- `_torch/moe/fused_moe/cutlass/w4a16_mxfp4.py:40`: the W4A16 MXFP4 CUTLASS MoE is sm90-only. `OVERRIDE_QUANT_ALGO=W4A16_MXFP4` on this card fails with
  "no MoE implementation can serve this layer … TrtllmCutlassW4a16Mxfp4Impl: sm_unsupported (supports SM90 only, got SM120)"
  (results/fairness/trt-20b-w4a16-pf256.log).

So the site compares pie W4A16 against TRT-LLM W4A8 on this card. TRT-LLM's numbers are real for TRT-LLM's default deployment, which
quantizes activations that the gpt-oss release keeps in bf16.

## Per-component breakdown (pf256, nsys)

Method:
- **pie:** `scripts/prof-pie-pb.sh`, nsys `--cuda-graph-trace=node` over `pie_bench`'s measured window (`--cuda-profiler-capture`).
- **TRT-LLM:** the same harness under nsys `--trace-fork-before-exec`, its kernels cut to the measured point's NVTX range (`scripts/kern_window.py`).
- **Classification:** components by kernel name (`scripts/kern_table.py`).
- **Equal tokens:** pie fed 512 tokens of its own rendering a prompt, so both engines prefill 131,072 tokens.
- **GPU saturation:** pie's kernel time is within 0.5% of its wall time, so the GPU is never idle, and the kernel deltas sum to the wall delta.

| component | 120b pie ms | 120b TRT-LLM ms | delta | 20b pie ms | 20b TRT-LLM ms | delta |
|---|---|---|---|---|---|---|
| MoE expert GEMMs (gate-up + down) | 3478 | 1818 | **+1660** | 2091 | 991 | **+1101** |
| MoE routing / act quantize / SwiGLU / combine / bias | 129 | 297 | −168 | 84 | 186 | −102 |
| dense GEMMs (qkv, o, router, head) | 727 | 677 | +50 | 482 | 450 | +32 |
| norms, residuals, rope, KV write | 166 | 140 | +26 | 109 | 89 | +20 |
| attention | 82 | 98 | −16 | 54 | 65 | −10 |
| other | 1 | 12 | −11 | 1 | 10 | −9 |
| **total kernel time** | **4582** | **3041** | **+1542** | **2822** | **1790** | **+1031** |
| wall (measured point) | 4604 | 3079 | +1525 | 2840 | 1868 | +972 |

- **The MoE:** expert GEMMs plus glue are +1,492 ms (97%) of 120b's delta and +999 ms (97%) of 20b's.
  - pie's expert GEMMs run 1.9× (120b) and 2.1× (20b) TRT-LLM's time for the same tokens.
  - pie's glue is cheaper: its GLU and bias are fused into the gate-up select, while TRT-LLM pays for a separate `doActivationKernel` (204 ms on 120b) and `expandInputRowsKernel` (61 ms).
- **Everything else at equal precision is +49 ms (120b) and +32 ms (20b), about 1% of wall:**
  - dense projections: cuBLAS `nvjet_sm120` vs TRT-LLM's `cutlass_80_tensorop_bf16_s16816gemm`, +7%;
  - norms/rope/KV write: `row_skeleton` vs `FusedAddRMSNorm` + `applyBiasRopeUpdateKVCacheV2`, +18%;
  - attention: pie's fa2 prefill is 16% faster than `fmha_v2_flash_attention_bf16_64_32_S_qkv_64_causal_sm120`.
- **Rates:** TRT-LLM's grouped GEMMs run ~517 TFLOP/s of expert work (7.17 GFLOP a token × 131,072 tokens in 1.82 s). pie's W4A16 selects run ~270 (3.48 s).
  cuBLAS's bf16 dense GEMMs run ~400 TFLOP/s in the same trace, so even a perfect W4A16 select would not reach TRT-LLM's W4A8 rate.

The kernel lists behind every number: results/prof/*_cuda_gpu_kern_sum.csv (pie), results/prof/trt-*-pf256_window.csv (TRT-LLM).

## Root cause

**pie (W4A16):**
- The gpt-oss release stores MXFP4 experts and bf16 everything else, and pie serves it as stored (deployments at `models/gpt_oss.poem:17-31`).
- Its wide selects run `select_mxfp4_mma` (`crates/kernels-cuda/src/linear/moe/select_fp4.rs:156`) on `moe_select_mxfp4_mma_staged<8,3,{3,0},false,8>` (`crates/kernels-cuda/kernels/linear/moe/select_fp4_sm80.cu:159`).
- Each warp dequantizes its e2m1 codes and ue8m0 scales to bf16 in registers (`crates/kernels-cuda/kernels/linear/moe_w4.cuh:268`, `compute_staged`), then multiplies them on bf16 `mma.sync.m16n8k16` (`moe_w4.cuh:330`).
- That caps the select at the bf16 tensor rate, minus the dequant ALU work: ~270 TFLOP/s.

**TRT-LLM (W4A8):**
- It quantizes the expert inputs to MXFP8 (`_torch/moe/fused_moe/cutlass/base.py:469-488`) and runs CUTLASS's SM120 TMA warp-specialized grouped GEMM on the block-scaled FP8×FP4 MMA (`w4a8_mxfp4_mxfp8.py:36-41`, `libth_common.so`).
- That MMA runs at twice the bf16 rate and needs no dequant: ~517 TFLOP/s.

**Things that are not the cause:**
- **The 8-warp pipe select:** it takes a K of whole 128s only (`select_fp4.rs`, the `pipe` condition), so gpt-oss's K = 2880 runs the staged kernel. I prototyped the pipe kernel with a zero-filled 64-code tail and register-prefetched scales (worktree only): it ran and was no faster (gate-up 1198 vs 1196 ms, 20b pf256; results/prof/pipe-20b-pf256*). Both kernels are bound by the same dequant + bf16 MMA.
- **Fire size:** 8192-row fires (TRT-LLM's `max_num_tokens`) instead of the profile's `fire_rows = 5120` give 20b pf256 46.9k vs 47.0k (results/explore).
- **The MXFP4 bf16 copies (#1401):** a Hopper path (`kernel.moe.mxfp4_bf16_from`, sm90 profiles only). sm_120 never takes it, and the 120b bank's bf16 copy (229 GB) could not fit anyway.
- **#1430:** the `select_fp4_sm120.cu` units serve releases that quantize rows (`crates/engine-cuda/src/dispatch/linear.rs:2054`). gpt-oss runs the same W4A16 selects at ee9098f60 and on main.

## A W4A8 gpt-oss path on sm_120a (decision for Felix, measurement only)

pending

## Other loads and related cells

pending

## Side findings

pending

## Playbook notes

pending
