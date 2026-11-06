# status: 20261106-1706-h100-sxm-x1-qwen3-6-27b-batching

- 17:21Z phase: setup. Models downloaded (Qwen3.6-27B, Qwen3.8-27B). Building pie main (dc6282d76), installing TRT-LLM 1.4.0rc2.post1 and vLLM 0.32.0 venvs.
- Site's TRT-LLM harness is not public (anton-mel/stats is Mac/Ollama only); I adapt the PRO 6000 run's LLM-API harness (trtllm_pbatch.py) to batching with KV reuse off.
- Early verdict ETA ~19:30Z. Hand-over ETA ~01:15Z.
