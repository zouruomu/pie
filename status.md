# status: 20261106-0910-pro6000-x1-gpt-oss-120b-pbatch

- 2026-11-06 09:35Z — phase: setup. Models downloaded (20b, 120b). Building pie engine (maturin, venv-pie), inferlet built.
  Installing TRT-LLM 1.4.0rc2.post1 (venv-trtllm) and vLLM 0.32.0 (venv-vllm).
- Site's TRT-LLM harness: not public (pie-project/benchmarks is private; anton-mel/stats has no TRT-LLM). Writing a cold-prompt
  TRT-LLM LLM-API harness that mirrors pie_bench's pbatch shape (same prompts file, 1 output token, KV reuse off; also measured with reuse on).
- ETA early verdict: ~11:40Z. ETA hand-over: ~19:10Z.
