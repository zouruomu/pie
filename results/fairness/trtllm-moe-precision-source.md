# TensorRT-LLM 1.4.0rc2.post1: what gpt-oss's MoE runs on sm_120 (source read of the installed wheel)
Paths relative to site-packages/tensorrt_llm/.
- `_torch/model_config.py:579-587`: GptOssForCausalLM MoE backend AUTO -> "TRTLLM" for 100<=sm<120, "TRITON" for sm90, else "CUTLASS" (sm_120 -> CUTLASS).
- `_torch/model_config.py:730-741` (get_mxfp4_quant_algo): sm>=100 and backend != TRITON -> QuantAlgo.W4A8_MXFP4_MXFP8; W4A16_MXFP4 only below sm100. No sm12x branch. Override only via env OVERRIDE_QUANT_ALGO (`:1094-1108`).
- `_torch/moe/fused_moe/cutlass/w4a8_mxfp4_mxfp8.py:36-41`: TrtllmCutlassW4a8Mxfp4Mxfp8Impl, sm_support {100,103,107,120,121}, quantize_input = mxfp8_quantize, use_mxfp8_act_scaling=True.
- `_torch/moe/fused_moe/cutlass/base.py:469-488`: torch.ops.trtllm.mxfp8_quantize(x) on every forward; `:371` torch.ops.trtllm.fused_moe (permute, FC1, gpt-oss SwiGLU, FC2+finalize).
- libs/libth_common.so symbols: tma_warp_specialized_generic_moe_gemm_kernelLauncher<Sm120, __nv_fp8_e4m3, __nv_fp4_e2m1, ..., 128_128_128_1_1_1> (block-scaled MMA, UE8M0 scales a 32).
- `_torch/moe/fused_moe/cutlass/w4a16_mxfp4.py:40`: the CUTLASS W4A16 MXFP4 kernel is sm90-only.
- Attention: TRTLLM attention backend (sinks require it, `_torch/models/modeling_gpt_oss.py:91-95`), qkv/out/router/embeddings excluded from quantization (bf16).
- Defaults: max_num_tokens 8192 (`llmapi/llm_args.py:5932`), enable_block_reuse True (`:4876`), tokens_per_block 32, chunked prefill off, CUDA graphs decode-only.
So on this card TRT-LLM's MoE is W4A8 (MXFP4 weights x MXFP8 e4m3 activations, a 32-element UE8M0 scale), pie's is W4A16 (bf16 activations).
