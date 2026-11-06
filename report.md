# RTX PRO 6000 Blackwell × gpt-oss-120b / gpt-oss-20b × prefill batch: pie vs TensorRT-LLM

Run 20261106-0910-pro6000-x1-gpt-oss-120b-pbatch. pie main dc6282d76; TensorRT-LLM 1.4.0rc2.post1; vLLM 0.32.0.
Card: NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 96 GB, 600 W cap), driver 595.91.07, CUDA 13.4.

## Summary

- **Both cells are red on main, and the gap is the MoE expert GEMMs, which TRT-LLM runs at W4A8.** At their best points, 20b runs pie 47.0k vs TRT-LLM 72.2k tok/s (−35%) and 120b runs 29.0k vs 43.4k (−33%).
  - The gap is −33…−35% from pf128 to pf1024 on both models, −33% at pf64, and −29…−31% at pf32.
  - At pf4/pf8, pie leads or ties on 120b (+18%, 0%) and trails by 9–10% on 20b.
- **TRT-LLM runs gpt-oss's experts at W4A8 on sm_120:** MXFP4 weights × MXFP8 (e4m3, one ue8m0 exponent a 32) activations, on a CUTLASS SM120 block-scaled grouped GEMM.
  - This is its only gpt-oss MoE path on sm_120. Forcing W4A16 fails with "TrtllmCutlassW4a16Mxfp4Impl supports SM90 only".
  - pie runs W4A16, which is what the release stores: bf16 activations on its register-dequant bf16 `mma.sync` selects.
- **At equal token counts, 97% of the gap is the MoE.** pf256, nsys:
  - 120b: +1,492 of +1,542 ms of kernel time is the MoE. The expert GEMMs alone are +1,660 ms, about 2× per token.
  - pie's MoE glue is 168 ms cheaper: its GLU and gate-up bias fuse into the select, and it needs no permute pass or activation quantizes.
  - Everything else (attention, dense projections, norms, rope/KV) totals +49 ms, about 1%.
- **At W4A16 pie is level with or ahead of vLLM 0.32 (MARLIN W4A16), the other W4A16 engine:**
  - 20b: pie leads at every pf (+4.3% to +30%).
  - 120b: pie leads at pf4–pf64 and is level from pf128 up (pf256 +0.2%, pf1024 −0.8%; pie n=5, vLLM n=3 alternating, spreads ≤1.9%).
- **The brief's 120b-specific problem does not reproduce, on main or at the site's commit.**
  - On this pod, ee9098f60 runs 28.7k tok/s at 120b pf256, with the same MoE kernels and per-component times as main (MoE total within 1%).
  - The site's 13.8k for that commit is not reproducible here. Its vLLM row also ran 24% low that day, and pie's 52% low; what happened on the site is unknown.
  - #1430 changes nothing for gpt-oss: both commits run the same W4A16 staged selects.
- **No fix PR:** per the orchestrator (note 3), changing gpt-oss's default MoE precision is out of scope. Nothing at W4A16 reaches the 4.5% bar: not against TRT-LLM outside the MoE (+1% of wall), not against vLLM, and not in the 20b pf16 "dip", which is noise. What a W4A8 gpt-oss path on sm_120 would take, and what it would gain, is below as a decision for Felix.

## Reproduced baseline (main dc6282d76)

**Setup:**
- Engines: pie embedded engine sha256 `22af6e94…b143` (recorded in every pie JSON), TRT-LLM 1.4.0rc2.post1 (PyTorch backend, LLM API), vLLM 0.32.0 (`scripts/bench/vllm_bench.py`).
- Prompts: `results/prompts-gptoss-512.json`, unique and cold, HF-rendered to exactly 512 tokens; 16 warmup prompts first, then N measured prompts offered at once; 1 output token, greedy.
- One engine at a time on an idle GPU. pie and vLLM run one process a point (as `pie_bench`/`vllm_bench` run). TRT-LLM runs a point list in one process after one warmup.
- prefill tok/s = prompt tokens / wall. Each cell gives the median, (min–max, n) over rounds.
- Rounds: three alternating at pf128–pf1024 for pie and TRT-LLM; three alternating pie/vLLM at the points within 4.5% (120b pf64/256/1024, 20b pf512); four for pie at 20b pf8/pf16; one elsewhere.
- TTFT is pie's latency p50 (one output token) and TRT-LLM's first-token p50.

**Clocks:**
- pie 120b pf1024 in-run (results/clocks/, 100 ms samples): over the measured window's 156 contiguous samples at ≥90% utilization, SM 2400 MHz median (min 2392), **600 W median, at the cap**.
- TRT-LLM at its large points: 2265–2332 MHz at the 600 W cap (per point in its JSONs).
- Both engines run at the power cap, and pie at the higher clock, so clocks are not the gap.


### gpt-oss-20b

| pf | pie | TRT-LLM | vLLM | pie vs TRT-LLM | pie vs vLLM | pie TTFT p50 ms | TRT-LLM TTFT p50 ms |
|---|---|---|---|---|---|---|---|
| 4 | 29.6k (n=1) | 32.6k (n=1) | 25.5k (n=1) | -9.3% | +16.1% | 46 | 63 |
| 8 | 39.8k (30.0-41.3, n=4) | 44.2k (n=1) | 30.7k (n=1) | -10.0% | +29.7% | 88 | 92 |
| 16 | 40.2k (38.1-43.5, n=4) | 54.7k (n=1) | 32.5k (n=1) | -26.4% | +24.0% | 129 | 149 |
| 32 | 45.0k (n=1) | 64.8k (n=1) | 41.3k (n=1) | -30.6% | +8.8% | 224 | 151 |
| 64 | 45.7k (n=1) | 67.6k (n=1) | 42.4k (n=1) | -32.5% | +7.8% | 333 | 272 |
| 128 | 46.2k (45.9-46.7, n=3) | 70.2k (69.0-70.7, n=3) | 43.5k (n=1) | -34.2% | +6.4% | 646 | 500 |
| 256 | 46.7k (46.0-47.0, n=3) | 71.5k (69.9-71.7, n=3) | 44.3k (n=1) | -34.7% | +5.4% | 1274 | 955 |
| 512 | 46.7k (46.4-47.0, n=5) | 72.1k (71.5-72.1, n=3) | 44.7k (44.7-44.8, n=3) | -35.3% | +4.3% | 2523 | 1879 |
| 1024 | 47.0k (46.9-47.0, n=3) | 72.2k (71.4-72.3, n=3) | 44.8k (n=1) | -35.0% | +4.8% | 4943 | 3723 |

prompt tokens a request: {'pie': [448], 'trt': [512], 'vllm': [512]}

### gpt-oss-120b

| pf | pie | TRT-LLM | vLLM | pie vs TRT-LLM | pie vs vLLM | pie TTFT p50 ms | TRT-LLM TTFT p50 ms |
|---|---|---|---|---|---|---|---|
| 4 | 20.7k (n=1) | 17.5k (n=1) | 14.6k (n=1) | +18.2% | +41.5% | 86 | 117 |
| 8 | 24.6k (n=1) | 24.7k (n=1) | 19.8k (n=1) | -0.3% | +24.2% | 144 | 166 |
| 16 | 26.7k (n=1) | 34.3k (n=1) | 22.7k (n=1) | -22.0% | +17.8% | 183 | 238 |
| 32 | 27.9k (n=1) | 39.1k (n=1) | 25.6k (n=1) | -28.5% | +9.2% | 351 | 245 |
| 64 | 27.8k (27.8-28.2, n=3) | 41.3k (n=1) | 27.4k (27.2-27.4, n=3) | -32.6% | +1.7% | 591 | 434 |
| 128 | 28.5k (28.5-28.8, n=3) | 42.4k (42.2-42.7, n=3) | 28.4k (n=1) | -32.8% | +0.6% | 1057 | 817 |
| 256 | 28.9k (28.5-29.0, n=5) | 43.2k (43.1-43.4, n=3) | 28.9k (28.9-28.9, n=3) | -33.0% | +0.2% | 2046 | 1566 |
| 512 | 29.0k (29.0-29.1, n=3) | 43.4k (43.3-43.5, n=3) | 29.1k (n=1) | -33.3% | -0.5% | 4065 | 3083 |
| 1024 | 29.0k (29.0-29.1, n=5) | 43.4k (43.3-43.6, n=3) | 29.3k (29.2-29.3, n=3) | -33.2% | -0.8% | 7947 | 6129 |

prompt tokens a request: {'pie': [448], 'trt': [512], 'vllm': [512]}

- pie prefills 448 tokens a prompt, TRT-LLM and vLLM 512 (see Fairness). At 512 of pie's own tokens, pie's tok/s is unchanged (results/equal-tokens: 120b pf256 29.0k, 20b 46.6k, one round each).
- Small points are noisy for pie at 0.1–0.2 s walls: four rounds at 20b pf8 span 30.0–41.3k, pf16 38.1–43.5k. Round 1's pf16 "dip" (38.1k below pf8's 41.3k) is not there in the medians (39.8k, 40.2k).
- **No regression since ea800efeb (the site's 20b commit).** The site's 20b pie number is 49.8k at ea800efeb, TTFT p50 188 ms, 6% above main's best here. I ran ea800efeb (engine sha256 `75392759…`) vs main (`22af6e94…`) at 20b pf32, pf256 and pf1024: 3 alternating rounds, the same prompts file, both engines prefilling 448 tokens a prompt (results/ea800e-ab/):

  | pf | ea800efeb | main | main vs ea800efeb |
  |---|---|---|---|
  | 32 | 43.0k (42.3–43.5, n=3) | 43.7k (43.7–44.0, n=3) | +1.7% |
  | 256 | 46.0k (45.7–46.3, n=3) | 46.6k (46.6–46.8, n=3) | +1.2% |
  | 1024 | 46.8k (46.4–46.9, n=3) | 46.8k (46.7–46.8, n=3) | −0.1% |

  There is no regression of 4.5% or more at any of these points; main is equal or faster. ea800efeb doesn't reach 49.8k on this pod either, so the site's 20b pie number doesn't reproduce here at the site's own commit.
- An 8192-row fire (`--max-forward-tokens 8192`, TRT-LLM's max_num_tokens) instead of the profile's `fire_rows = 5120`:
  - 20b: pf256 46.9k (same), pf16 36.9k vs 43.1k, pf32 41.4k vs 43.8k: slower at the small points;
  - 120b pf16: 27.3k vs 26.7k, one round each.
  The profile's split stays.

## Fairness of TensorRT-LLM's numbers, and the precision each engine ran

The site's TRT-LLM harness is not public: pie-project/benchmarks is private, and anton-mel/stats has no TRT-LLM. So I measured it with
`results/scripts/trtllm_pbatch.py`, an LLM-API harness that mirrors `pie_bench.py tput`:
- the same prompts file, with 16 warmup prompts first and the same system prompt, rendered by HF's chat template to exactly 512 tokens;
- every request offered at t=0 (concurrency = N);
- `max_tokens=1`, greedy;
- prefill tok/s = prompt tokens / wall;
- all other settings at TRT-LLM's defaults, except **KV block reuse off** (cold prompts, as pie and vLLM run them; TRT-LLM's default is on).

- **Reproduces the site:** 20b pf256 71.5k (site 71.3k), 120b pf256 43.2k (site 42.4k, +2%), with reuse off. So the site's TRT-LLM lead is not prefix reuse.
  With reuse on, the 80 harmony tokens the prompts share (2 blocks of 32) would be skipped.
- **Same prompts, same output, best point:** 512-token prompts, 1 output token. TRT-LLM's best point is pf512–pf1024 (72.2k / 43.4k).
- **Clocks:** TRT-LLM runs at the 600 W cap at its large points, with SM clocks at 2265–2317 MHz (`clocks_before/after` per point in its JSON).
  pie, sampled in-run at 120b pf1024 (results/clocks/), also sits at the 600 W cap, at 2400 MHz. The per-run `.clocks` files are sampled before and after each process, not in-run.
- **Tokens actually prefilled:** pie 448 a prompt, TRT-LLM and vLLM 512.
  - pie's harmony template (`crates/chat-template/src/harmony.rs:142-147`) sends the system prompt as a developer message. It has no system message: no identity, knowledge cutoff, date, reasoning effort or valid-channels lines, which is 64 tokens.
  - Each engine's tok/s is over its own count. At equal counts (pie fed `prompts-gptoss-576.json`, which pie renders to exactly 512 tokens), pie runs the same per-token rate: 120b pf256 29.0k (vs 28.9k at 448), 20b 46.6k (vs 46.7k), from results/equal-tokens. The nsys-profiled equal-token runs (results/prof/*-t512) are 28.5k and 46.1k.

**Precision each engine ran:**

| engine | MoE experts | activations into the experts | attention / dense |
|---|---|---|---|
| pie dc6282d7 (deployment `gptoss-*-bf16-mxfp4-kv-bf16`) | MXFP4 (as released) | bf16 (W4A16), register-dequant bf16 `mma.sync` (`moe_select_mxfp4_mma_staged`) | bf16 |
| vLLM 0.32.0 | MXFP4 | bf16 (W4A16), MARLIN (`mxfp4.py:709` "Using 'MARLIN' Mxfp4 MoE backend") | bf16, FlashInfer |
| TRT-LLM 1.4.0rc2.post1 | MXFP4 | **MXFP8**: e4m3, a ue8m0 exponent a 32 (W4A8) | bf16 (qkv, o, router, embeddings excluded from quantization) |

TRT-LLM source, relative to site-packages/tensorrt_llm (full notes: results/fairness/trtllm-moe-precision-source.md):
- `_torch/model_config.py:579-587`: gpt-oss MoE backend AUTO → CUTLASS on sm_120.
- `_torch/model_config.py:729-741`: sm ≥ 100 and backend != TRITON → `QuantAlgo.W4A8_MXFP4_MXFP8`. W4A16 is chosen only below sm100.
- `_torch/moe/fused_moe/cutlass/w4a8_mxfp4_mxfp8.py:36-41`: `quantize_input = quantize_input_mxfp8` (which calls `torch.ops.trtllm.mxfp8_quantize`, `base.py:484-486`), sm {100, 103, 107, 120, 121}.
- `_torch/moe/fused_moe/cutlass/base.py:469-488`: the activations quantized every forward; `:371` `torch.ops.trtllm.fused_moe`.
- `libs/libth_common.so`: `tma_warp_specialized_generic_moe_gemm_kernelLauncher<Sm120, __nv_fp8_e4m3, __nv_fp4_e2m1, …, 128_128_128_1_1_1>`.
- `_torch/moe/fused_moe/cutlass/w4a16_mxfp4.py:40`: the W4A16 MXFP4 CUTLASS MoE is sm90-only. `OVERRIDE_QUANT_ALGO=W4A16_MXFP4` on this card fails with
  "no MoE implementation can serve this layer … TrtllmCutlassW4a16Mxfp4Impl: sm_unsupported (supports SM90 only, got SM120)"
  (results/fairness/trt-20b-w4a16-pf256.log). TRT-LLM's other MXFP4-capable MoE, `TritonFusedMoE`, is SM90-only (`_torch/moe/fused_moe/fused_moe_triton.py:1611`).
  The b12x CuteDSL MoEs are NVFP4-only.

So the site compares pie W4A16 against TRT-LLM W4A8 on this card. TRT-LLM's numbers are real for TRT-LLM's default deployment, which
quantizes activations that the gpt-oss release keeps in bf16.

## Per-component breakdown (pf256, nsys)

Method:
- **pie:** `results/scripts/prof-pie-pb.sh`, nsys `--cuda-graph-trace=node` over `pie_bench`'s measured window (`--cuda-profiler-capture`).
- **TRT-LLM:** the same harness under nsys `--trace-fork-before-exec`, its kernels cut to the measured point's NVTX range (`results/scripts/kern_window.py`).
- **Classification:** components by kernel name (`results/scripts/kern_table.py`).
- **Equal tokens:** pie fed 512 tokens of its own rendering a prompt, so both engines prefill 131,072 tokens.
- **GPU saturation:** pie's kernel time is within 0.5% (120b) and 0.7% (20b) of its wall time.
  - On 120b the kernel delta (+1542 ms) matches the wall delta (+1525).
  - On 20b TRT-LLM's GPU idles 78 ms of its 1868, so the kernel delta (+1031) overstates the wall delta (+972) by ~60 ms.

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
  - pie's glue is cheaper. Its GLU and gate-up bias are fused into the select, and at W4A16 it needs no permute or activation quantize; it still runs a separate `moe_bias_sum` for the down bias.
  - TRT-LLM pays for `doActivationKernel` (SwiGLU + FP8 quantize, 204 ms on 120b) and `expandInputRowsKernel` (permute, 61 ms).
- **Everything else at equal precision is +49 ms (120b) and +32 ms (20b), about 1% of wall:**
  - dense projections: cuBLAS `nvjet_sm120` vs TRT-LLM's `cutlass_80_tensorop_bf16_s16816gemm`, +7%;
  - norms/rope/KV write: `row_skeleton` vs `FusedAddRMSNorm` + `applyBiasRopeUpdateKVCacheV2`, +18%;
  - attention: pie's fa2 prefill takes 16% less time than `fmha_v2_flash_attention_bf16_64_32_S_qkv_64_*_sm120`.
- **Rates:** TRT-LLM's grouped GEMMs run ~517 TFLOP/s of expert work (7.17 GFLOP a token × 131,072 tokens in 1.82 s). pie's W4A16 selects run ~270 (3.48 s).
  cuBLAS's bf16 dense GEMMs run ~350–390 TFLOP/s in the same trace: 2.54e14 FLOP in the 727 ms component; by kernel, ~360 for the qkv-sized nvjet and ~390 for the o-sized one.
  So even a W4A16 select at cuBLAS's bf16 rate would stay well below TRT-LLM's W4A8 517.

The kernel lists behind every number: results/prof/*_cuda_gpu_kern_sum.csv (pie), results/prof/trt-*-pf256_window.csv (TRT-LLM).

## Root cause

**pie (W4A16):**
- The gpt-oss release stores MXFP4 experts and bf16 everything else, and pie serves it as stored: deployment `gptoss-{20b,120b}-bf16-mxfp4-kv-bf16`, `models/gpt_oss.poem:20,24`.
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
- **#1430:** the `select_fp4_sm120.cu` units serve releases that quantize rows (`crates/engine-cuda/src/dispatch/linear.rs:2054`, the cuda `select_act_quant`). gpt-oss runs the same W4A16 staged selects at ee9098f60 and on main. main's trace also shows a small `moe_select_mxfp4_mma<4,1,…>` variant (42 ms) ee9098f60's does not; the MoE totals are within 1%.

## A W4A8 gpt-oss path on sm_120a (decision for Felix, measurement only)

Orchestrator note 3 scoped a default-precision change out of this PR. This section is decision material only. I built no W4A8 gpt-oss kernel.

**pie's own sm_120a W4A8 select is already near TRT-LLM's rate.** I timed `moe_select_mxfp4_w4a8_ws<3>` (TMA + warp-specialized, as the PRO 6000 profile runs it) with an ignored timing test added in a scratch worktree (`w4a8_select_timing_gptoss`, results/scripts/w4a8-measurement-only.diff, not part of any PR).
- Shape: gpt-oss-120b, a 5120-token fire, 128 experts, top 4, K padded 2880 → 3072, MXFP8 rows in 32s, synthetic balanced routes. Source: results/w4a8/w4a8-gptoss-timing*.

| leg | kernel time, median of 5 launches | TFLOP/s at K = 3072 | useful TFLOP/s at gpt-oss's K = 2880 |
|---|---|---|---|
| gate-up (N = 5760) | 1.41 ms | 514 | 481 |
| down (N = 2880) | 0.74 ms | 490 | 459 |
| `quantize_rows_mxfp8_routed<32>` | 0.040 ms (gate-up input), 0.114 ms (down input), median a launch | | |

**Projections**, on the clean equal-token pf256 walls (120b 4.523 s, 20b 2.811 s for 131,072 tokens; results/equal-tokens), replacing the profiled W4A16 expert-GEMM time with W4A8 time:
- **120b** (921.6 layer-fires of 5120 rows; W4A16 3.77 ms each, measured W4A8 2.31 ms with both quantizes, ~2.17 ms at the true K):

  | case | wall | tok/s |
  |---|---|---|
  | padded K, unfused GLU/bias | ~3.38 s | **~38.8k** |
  | true K, unfused | ~3.25 s | ~40.4k |
  | true K, fused epilogue | ~3.05 s | **~43.0k** |

  That is vs TRT-LLM's 43.2k at pf256. The unfused GLU/bias pass is ~0.2 s: the routes × 5760 product read again and the GLU written, ~354 MB a layer-fire.
- **20b** (not timed at its own shape, 32 experts): at the 120b-shape rates above plus the quantizes, the expert GEMMs would take ~1.42 s instead of 2.09 s. That gives **~58–61k** (unfused / fused) vs TRT-LLM's 71.5k, about half the gap.
  - TRT-LLM's own 20b grouped GEMMs run ~632 TFLOP/s against its ~517 on 120b: 4× the routes an expert give bigger GEMMs. pie's kernel would likely gain similarly, which this estimate does not include.
  - At TRT-LLM's own rates exactly, the projections are 120b ~43.6k and 20b ~72.6k. That is the ceiling if pie's W4A8 GEMMs matched TRT-LLM's.
- Synthetic balanced routes flatter the kernel somewhat relative to real routing.

So a W4A8 path would close most or all of the 120b gap and at least half of 20b's. Closing 20b fully would also need the W4A8 select tuned at 32 experts.

**What it needs, with estimates:**
1. **K = 2880.** The sm_120a W4A8 selects take only whole 128s:
   - `w4a8_takes`, `select_fp4.rs:413-421`;
   - `chunks = k >> 7` in `moe_select_mxfp4_w4a8_ws` (`select_fp4_sm120.cu`);
   - the routed quantize's chunk-major exponent words, `quantize_rows_mxfp8_routed`.

   The x and w boxes can rely on TMA's zero fill past the tensor's extent. The weight-scale loads are 8-byte `uint2` loads at `row * (k / 32)`, misaligned at k / 32 = 90, so they need narrower loads. The last chunk's exponent bytes must be zeroed, not left uninitialized: e8m0 0xFF is NaN. Plus `moe_select.rs` cases at K = 2880. **~4–6 h.**
2. **Gate-up epilogue.** gpt-oss's gate-up needs per-expert biases and SwiGLU clamped at 7 with alpha 1.702. The W4A8 select passes `bias = None`, GLU code 0 (`select_fp4.rs:453`).
   - Unfused, it can run the existing `swiglu_clamp_alpha` (`crates/kernels-cuda/src/linear/glu.rs:116`) and a bias pass, at the cost of an extra pass over the routes × 5760 product: **~2 h**.
   - Fused into the ws kernel's epilogue, as pie's W4A16 select does: **~4–6 h**.
3. **Precision plumbing.** A gpt-oss deployment `weights = [dtype.bf16, dtype.mxfp4a8g32]` (`models/gpt_oss.poem:17-31`). `mxfp4a8g32` already exists (`crates/poem/src/declare.rs:847-850`, in `Precision::ACTED` from :818).
   - Today the loader refuses a declared activation precision over the stock checkpoint: "stores its weights at [Bf16, Mxfp4] and the config asks for [Bf16, Mxfp4 a8g32]; import the checkpoint at that precision" (`crates/runtime/src/engine/load.rs:146`). I tried it on a worktree build.
   - Either the import path or accepting the declared act precision over an MXFP4 bank: **~1–2 h**.
   - Policy: the site measures pie's default deployment, so the cells change only if this becomes gpt-oss's default on sm_120, which is the precision decision itself. Opt-in only, the site would not see it.
4. **Quality evidence** (note 2): teacher-forced KL and top-1 agreement vs the W4A16 path over a few hundred tokens, greedy agreement on raw and batching runs, a short task-accuracy check: **~3 h**.
5. **Verification:** curve, cross-model, GPU suite: **~3–4 h.**

**Total: ~15–21 h**, about two to three working days. Without the fused epilogue (item 2 unfused), ~13–17 h, and part of the win goes to the extra pass.

The W4A8 select is gated at 256 routes a fire (`kernel.moe.mxfp4_w4a8_from = 256` in the PRO 6000 profile; `w4a8_takes` counts the fire's routes):
- single-request decode and fires of fewer than 64 tokens would stay W4A16;
- prefill and large-batch decode steps would go W4A8.

So a W4A8 deployment makes gpt-oss's numerics depend on the fire width, and the quality evidence has to cover both.

## Other loads and related cells

- **120b batching (related red cell, site −30%):** one measurement each at c1024 (1024 requests offered at once, 128 prompt tokens of each engine's own rendering, 128 out; results/batching/).
  - pie: 131,072 output tokens in 16.0 s, **8,185 out tok/s**. Its tput mode records end-to-end latency (p50 9.98 s), not TTFT.
  - TRT-LLM: 18.4 s, **7,119 out tok/s**, TTFT p50 1.85 s.
  - In this shape (n=1) pie leads by 15%. I don't know the site's batching request shape, so this neither explains nor rules out the site's −30%.
  - The site's pie TTFT p50 of 20.9 s against TRT-LLM's 0.53 s points at requests waiting for admission rather than slower steps. That is the area of the admission cap from #1368 that the concurrent A100 run is lifting where pools seat to the end. It is a hypothesis; per note 4 I did not chase it.
- **No change was made,** so there are no fix-vs-base checks (token equality, cross-model, memory fit, GPU suite) to report.

## Self-review

An independent fresh-context review (results/review-1.md) recomputed every number from the raw files and opened every `file:line` cited.
- Confirmed: the tables and the 97% attribution.
- Fixed in this version: stale round-1 numbers in the summary, pie's in-run power (600 W, not 561), the effort totals, the W4A8 projections (now on the clean walls with pie's measured quantize, 20b at a stated rate), and over-strong conclusions on the site run, ea800efeb and batching.

## Side findings

- **The site's 2026-11-03 rows for 120b do not reproduce for either engine:**
  - The site has pie 13.8k (ee9098f60) and vLLM 22.3k. On this pod, ee9098f60 runs 28.7k at pf256 and vLLM 0.32 29.3k at its best point (results/baseline/ee9098-*, vllm-120b-*).
  - vLLM's row is 0.76× of its number here, pie's 0.48×, so pie's site row has a further unexplained shortfall. The site's 120b gap to vLLM (−38%) and the 0.28 120b/20b ratio come from that run.
  - pie's 120b/20b ratio here is 0.62, TRT-LLM's 0.60.
- **pie's harmony prompt is 64 tokens shorter than HF's** (`crates/chat-template/src/harmony.rs:142-147`).
  - pie sends the system prompt as a developer message with no system message: no identity, knowledge cutoff, current date, "Reasoning: medium", or valid-channels lines.
  - Every engine's tok/s on the site is over a different token count, and TTFTs compare prompts of different lengths: pie 448 vs 512 here. If the site's batching feeds a 128-HF-token file, pie prefills 64. My batching run fed pie 128 of its own tokens.
  - Whether the model should see that preamble is a template-fidelity question for the maintainers. It is not a throughput issue: pie's tok/s is the same at 512 tokens of its own rendering.
- **TRT-LLM's KV block reuse is on by default.** The harmony prompts share 80 tokens (2 blocks of 32). My cold runs reproduce the site's numbers, so either the site turned it off or it made no visible difference. Harness authors should turn it off explicitly (`KvCacheConfig(enable_block_reuse=False)`).
- **The 8-warp pipe select's K-tail prototype is not worth shipping.** It took gpt-oss's K = 2880 by zero-filling the last half chunk and reading scales a chunk ahead from registers, and it ran at the same speed as the staged select (results/prof/pipe-20b-pf256*). Both are dequant + bf16 `mma.sync` bound.
- **RTX 5090 and RTX PRO 4500 very likely share the cause** (a hypothesis from code, not measured):
  - On every sm_120 card, `select_mxfp4_mma` (`select_fp4.rs:156`) dispatches gpt-oss's wide bf16-activation selects to the same W4A16 staged kernel: no tcgen05 convert path, no Hopper wgmma, not the Ada run kernel. Their profiles (`crates/engine-cuda/profiles/sm120/`) change only thresholds. I read this from the code; I did not trace those cards.
  - TRT-LLM's AUTO path is W4A8 on every sm_120/121 card (`model_config.py:579-587, 729-741`; `SmSupport {…, 120, 121}`).
  - The 5090's profile differs in thresholds (`moe.mxfp4_mma_from = 21`, `moe.tc5`). Why its gap is larger (pbatch −44%) is not measured here.
- **The 20b pf16 dip** was round-to-round noise at 0.1–0.2 s walls (see the baseline notes).

## Playbook notes

- **Timestamps:** take every status time from `date -u`. Elapsed time is easy to misjudge while waiting on queues; the orchestrator caught one status line 20 minutes off.
- **TRT-LLM 1.4 setup on a fresh pod:**
  - needs `libopenmpi-dev`;
  - must load from the local HF snapshot path: given the repo id, it re-downloads and failed on the files the `hf download` excluded;
  - `OVERRIDE_QUANT_ALGO` is the only switch for gpt-oss's MoE precision;
  - for nsys, trace child processes too (`--trace-fork-before-exec=true`) and cut the kernels to an NVTX range pushed around the measured point (`results/scripts/kern_window.py`).
- **TRT-LLM kernel names:** classify `cutlass::gemm::GroupProblemShape` kernels as the MoE. A naive "gemm" rule files them under dense GEMMs.
- **Equal tokens:** for harmony models (gpt-oss), compare at equal prompt-token counts. pie's prompt is 64 tokens shorter than HF's; build a pie-specific prompts file (HF length + 64).
- **Engines from other commits:** an engine at another commit builds in ~2 minutes in a git worktree with its own `CARGO_TARGET_DIR` (`cargo build --release --lib` in `packages/python/server`, then copy `libpie_engine.so` to the worktree's `python/pie/_engine.abi3.so`). Point `PYTHONPATH` at the worktree's `python` dir to swap engines without touching the main build. This answered the site-commit question (ee9098f60) in minutes.
- **zsh:** it runs `&` jobs at nice 5 (BG_NICE). Foreground tool calls stop at 10 minutes, so queue GPU work in scripts and wait with a background `until` loop.
