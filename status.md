# status: 20261106-1706-h100-sxm-x1-qwen3-6-27b-batching

- 17:34Z phase: reproduce (Qwen3.6-27B first). pie main dc6282d76 built; TRT-LLM 1.4.0rc2.post1 and vLLM 0.32.0 installed.
- **Fairness finding (TRT-LLM):** TRT-LLM 1.4 allocates the Qwen3.5-family GDN recurrent state in **bf16**, deliberately ignoring the checkpoint's `mamba_ssm_dtype: float32` (`tensorrt_llm/_torch/pyexecutor/config_utils.py:300-325`, its docstring: fp32 "doubles state memory traffic (~20% serving throughput loss)"). vLLM 0.32 honors fp32 (`vllm/model_executor/models/config.py:853-875`). pie: checking. This halves TRT-LLM's state memory per lane (~78.5 MB vs ~157 MB) on a card where the state slots bound the seats. Measuring TRT-LLM with `mamba_ssm_cache_dtype=float32` next.
- TRT-LLM at its defaults does not start on this card: max_batch_size 2048 needs 160 GB of state slots; max_batch_size 256 OOMs at warmup/graph capture (mem fraction 0.9, 0.85) or does not fit the slots (0.75). Running max_batch_size 128 now. The site's TRT settings are unknown (its harness is private).
- Running: TRT-LLM q36 sweep (mb128), then pie main q36 sweep. Early verdict ETA ~19:30Z; hand-over ETA ~01:15Z.
