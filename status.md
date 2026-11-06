# status: 20261106-0910-pro6000-x1-gpt-oss-120b-pbatch

## Early findings (09:45Z, verdict pending the sweep)
- pf256 smoke on main dc6282d7 (1 round each, cold prompts): 20b pie 46.8k vs TRT-LLM 71.5k tok/s (-35%);
  120b pie 28.9k vs TRT-LLM 43.3k (-33%). pie 120b on main is 2.1x the site's 13.8k: the 120b-specific
  problem the brief suspected is gone on main (120b/20b now 0.62, TRT-LLM's 0.61). The remaining gap is common to both models.
- TRT-LLM's numbers reproduce with KV block reuse OFF (71.5k vs site 71.3k; 43.3k vs 42.4k): not a prefix-reuse artifact.
- **Precision differs.** TRT-LLM 1.4.0rc2.post1 runs gpt-oss's MoE on sm_120 as W4A8: MXFP4 weights x MXFP8 (e4m3, ue8m0 a 32)
  activations on a CUTLASS SM120 block-scaled grouped GEMM (`_torch/model_config.py:730-741`, `_torch/moe/fused_moe/cutlass/w4a8_mxfp4_mxfp8.py:36-41`).
  It has no W4A16 MXFP4 MoE on sm_120 (OVERRIDE_QUANT_ALGO=W4A16_MXFP4 fails: "TrtllmCutlassW4a16Mxfp4Impl supports SM90 only").
  pie runs W4A16 (bf16 activations). Details: results/fairness/.
- pie renders 448 tokens a prompt for the same prompt where HF/TRT-LLM render 512 (pie's harmony template drops the 64-token
  "You are ChatGPT..." system preamble). tok/s is per engine's own count; noted as a harness side finding.
- pie has an opt-in `mxfp4a8g32` precision (MXFP4 x e4m3 a 32, = TRT-LLM's W4A8) and an sm_120a W4A8 select, but gpt-oss declares no such
  deployment and the sm_120a W4A8 kernel needs K % 128 == 0 (gpt-oss K = 2880 = 22.5 x 128; `select_fp4.rs:417`, `select_fp4_sm120.cu` chunks = k >> 7).

## Running
- 09:32Z: baseline sweep round 1 (pie then TRT-LLM, 20b then 120b, pf4..pf1024). ETA ~10:30Z.
- building an engine with an opt-in gpt-oss W4A8 deployment in a worktree (measurement only, not a proposed change yet).

## Question for the orchestrator (decision needed before a fix)
The gap looks like precision (W4A8 vs W4A16) at large M. Closing it on the site needs pie's default gpt-oss MoE precision on sm_120 to change
to W4A8 on wide selects (what TRT-LLM does by default), or the site to compare at equal precision. I will quantify both (pie W4A8 vs TRT-LLM,
pie W4A16 vs vLLM W4A16) before proposing; tell me if a default-precision change is out of scope.

ETA early verdict ~10:45Z; hand-over ~19:00Z.
