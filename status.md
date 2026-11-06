# status: 20261106-0910-pro6000-x1-gpt-oss-120b-pbatch

## EARLY VERDICT (10:25Z): both cells RED on main dc6282d7, the gap is TRT-LLM's W4A8 MoE precision on sm_120
Round 1 of the full pf sweep (cold prompts, 1 output token, one engine at a time; rounds 2-3 at pf128-1024 running).
SM clocks during TRT-LLM's large points 2265-2317 MHz at the 600 W cap; pie's run-time clocks are recorded per run in results/baseline/*.clocks
(they are sampled before/after the process, not during).

| pf | 20b pie | 20b TRT-LLM | gap | 120b pie | 120b TRT-LLM | 120b vLLM | gap vs TRT |
|---|---|---|---|---|---|---|---|
| 4 | 29.6k | 32.6k | -9% | 20.7k | 17.5k | | +18% |
| 8 | 41.3k | 44.2k | -7% | 24.6k | 24.7k | | 0% |
| 16 | 38.1k | 54.7k | -30% | 26.7k | 34.3k | | -22% |
| 32 | 45.0k | 64.8k | -31% | 27.9k | 39.1k | | -29% |
| 64 | 45.7k | 67.6k | -32% | 27.8k | 41.3k | | -33% |
| 128 | 46.2k | 69.0k | -33% | 28.8k | 42.7k | | -33% |
| 256 | 47.0k | 69.9k | -33% | 29.0k | 43.2k | 28.9k | -33% |
| 512 | 46.7k | **71.5k** | -35% | 29.0k | **43.4k** | | -33% |
| 1024 | 46.9k | 71.4k | -34% | (pending r1 file) | 43.4k | | |

Best points: 20b pie 47.0k vs TRT-LLM 71.5k (-34%); 120b pie 29.0k vs TRT-LLM 43.4k (-33%). The site's TRT-LLM numbers reproduce (71.3k, 42.4k).
pie 120b on main is 2.1x the site's 13.8k (ee9098f60): the brief's 120b-specific problem is gone on main (checking which commit, ee9098f60 built).

**Why (attribution so far, pie 120b pf256 nsys):** 76% of pie's GPU time is the MXFP4 expert selects
(`moe_select_mxfp4_mma_staged`, register-dequant bf16 mma.sync, ~270 TFLOP/s; cuBLAS's bf16 dense GEMMs reach ~400 TFLOP/s on this card).
TRT-LLM 1.4.0rc2.post1 runs gpt-oss's MoE on sm_120 as **W4A8** (MXFP4 weights x MXFP8 e4m3 activations, CUTLASS SM120 block-scaled
grouped GEMM, FP8-rate tensor cores, no dequant): `_torch/model_config.py:730-741`. It has no W4A16 MXFP4 MoE on sm_120 (forcing it fails).
At equal precision pie is level: vLLM 0.32 (MARLIN W4A16) 120b pf256 = 28.9k vs pie 29.0k.
A W4A16 kernel can't close a 1.5x gap: the 8-warp pipe select with gpt-oss's K tail (prototyped, it ran) is no faster (1198 vs 1196 ms gate-up).
An 8192-row fire (TRT-LLM's max_num_tokens) instead of the profile's 5120 changes nothing (46.9k vs 47.0k).

**What closing it needs (decision for the orchestrator / Felix):** a W4A8 gpt-oss MoE on sm_120a, i.e. MXFP8 activations, which changes numerics.
pie's existing sm_120a W4A8 select (`moe_select_mxfp4_w4a8_ws`, used for DeepSeek-V4/MiMo releases that ask for e4m3) does not take gpt-oss:
K must be whole 128s (gpt-oss K = 2880), no fused GLU/bias/clamped SwiGLU, and pie refuses `weight_precision = bf16+mxfp4a8g32` for the stock
checkpoint ("import the checkpoint at that precision"). That is new kernel work plus a precision-policy change (pie serves gpt-oss at what the release
stores; TRT-LLM quantizes activations by default on Blackwell). Per the brief, I plan to hand over the attribution with the measured size of the
win (TRT-LLM's MoE kernel time vs pie's, from nsys next) rather than a half-built kernel, unless you want a W4A8 gpt-oss path attempted.


## Attribution (11:05Z), gpt-oss-120b pf256, nsys over the measured window (pie: whole capture; TRT-LLM: its NVTX-marked point)
pie wall 3.95 s for 114,688 prompt tokens (448 a prompt), TRT-LLM 3.08 s for 131,072 (512 a prompt). Kernel time by component:

| component | pie ms | TRT-LLM ms | pie ms / 1k tok | TRT-LLM ms / 1k tok |
|---|---|---|---|---|
| MoE expert GEMMs (gate-up + down) | 3083 | 1818 | 26.9 | 13.9 |
| MoE routing / act quantize / SwiGLU / combine / bias | 110 | 297 | 0.96 | 2.27 |
| dense GEMMs (qkv, o, router, head) | 640 | 677 | 5.58 | 5.16 |
| norms, residuals, rope, KV write | 141 | 140 | 1.23 | 1.07 |
| attention | 69 | 98 | 0.60 | 0.74 |
| other | 1 | 12 | | |
| total kernel time | 4044 | 3041 | 35.3 | 23.2 |

The whole gap is the expert GEMMs: pie's W4A16 register-dequant bf16 mma.sync selects vs TRT-LLM's W4A8 CUTLASS SM120 grouped GEMMs
(MXFP8 x MXFP4 block-scaled MMA), 1.94x per token. Everything else is within ~0.6 ms / 1k tokens (dense +0.4, norms +0.16; pie's MoE glue
is cheaper because its GLU and bias fuse into the select). So at W4A16 nothing besides the MoE GEMM precision accounts for the gap.
Equal-token traces (pie at 512 of its own tokens) are queued to confirm.

### Equal-token attribution (11:20Z): pie at 512 prompt tokens of its own rendering (prompts-gptoss-576.json), pf256, 131,072 tokens each

| component | 120b pie ms | 120b TRT-LLM ms | delta | 20b pie ms | 20b TRT-LLM ms | delta |
|---|---|---|---|---|---|---|
| MoE expert GEMMs (gate-up + down) | 3478 | 1818 | +1660 | 2091 | 991 | +1101 |
| MoE routing / act quantize / SwiGLU / combine / bias | 129 | 297 | -168 | 84 | 186 | -102 |
| dense GEMMs (qkv, o, router, head) | 727 | 677 | +50 | 482 | 450 | +32 |
| norms, residuals, rope, KV write | 166 | 140 | +26 | 109 | 89 | +20 |
| attention | 82 | 98 | -16 | 54 | 65 | -10 |
| other | 1 | 12 | -11 | 1 | 10 | -9 |
| total kernel time | 4582 | 3041 | +1542 | 2822 | 1790 | +1031 |
| wall | 4604 | 3079 | +1525 | 2840 | 1868 | +972 |

The MoE (GEMMs + glue) is +1492 of +1542 ms (97%) on 120b and +999 of +1031 (97%) on 20b. The rest, at equal precision, is +49 / +32 ms (~1%):
nothing at W4A16 near the 4.5% bar. TRT-LLM's W4A8 grouped GEMMs run ~517 TFLOP/s of expert work; pie's W4A16 selects ~267.
- ee9098f60 (the site's 120b commit) on this pod: 28.7k tok/s at pf256, same kernels and component times as main (results/prof/ee9098-*):
  the site's 13.8k was that site run, not pie. #1430 changes nothing for gpt-oss (same W4A16 staged selects at both commits).
- pie at 512 tokens a prompt: 120b 28.5k, 20b 46.1k at pf256 (vs 29.0k / 46.7k at 448): token count is not the gap.

## Fairness notes
- TRT-LLM reproduced with KV block reuse OFF (cold); its default is ON (80 shared harmony tokens = 2 blocks of 32 a prompt would be reused).
- pie renders 448 tokens a prompt where HF/TRT-LLM/vLLM render 512 (pie's harmony template has no "You are ChatGPT..." system preamble):
  each engine's tok/s is over its own count. Side finding for the harness.
- Precision: pie W4A16 (bf16 activations); vLLM 0.32 MARLIN W4A16; TRT-LLM W4A8 MXFP4xMXFP8. Attention/dense in bf16 in all three.

## Orchestrator notes
- note-1 (09:30Z): acknowledged. Times from `date -u`. #1430: the sm_120a W4A8 selects in `select_fp4_sm120.cu` are reached only by releases that
  quantize rows (`dispatch/linear.rs:2063-2108`) and need K % 128 == 0; gpt-oss runs W4A16 `moe_select_mxfp4_mma_staged` on main (nsys), so #1430
  cannot change its selects. ee9098f60 kernel list: pending (built, measuring after rounds 2-3).

- note-2 (09:42Z), note-3 (10:05Z): acknowledged. Default-precision change out of scope; W4A8 measurement only (<=2 h, hours estimate for the
  K = 2880 tail; my worktree's W4A8 deployment is refused for the stock checkpoint, so no pie-W4A8 number without an import/kernel). Next: the
  per-component split pie vs TRT-LLM at pf256 (MoE vs the rest), pie vs vLLM 0.32 over the whole sweep for both models, and equal-token runs
  (pie at 512 prompt tokens of its own rendering).

## Running
- rounds 2-3 done (3 rounds at pf128-1024, spread <2%). Running: vLLM 0.32 sweep both models (~11:20-12:00Z), then ea800efeb vs main at 20b pf8-64
  (site's 20b pie 49.8k is above main's best 47.0k: checking for a small-pf regression), pf16/pf32 fire-split A/B (20b dips at pf16),
  120b batching c64/c256/c1024 vs TRT-LLM, and pie's own sm_120a W4A8 select timed at gpt-oss-120b's shape (K padded to 3072). ETA ~13:15Z.
- ETA hand-over ~19:10Z (earlier if the decision is attribution-only).
