# Note 4 (orchestrator, 11:25Z): wrap up

The equal-token attribution settles it: 97% of the gap is the expert GEMM precision, and nothing at W4A16 is near the bar. Wrap up and hand over by about 13:30Z:
1. pie vs vLLM 0.32 over the whole pf sweep for both models: 1 round, plus 3 alternating rounds at any point within 4.5% either way. Look in particular at pie 20b's dip at pf16 (38.1k, below its pf8 41.3k). If vLLM or SGLang lead by ≥4.5% anywhere at equal precision, say so and we decide then.
2. The W4A8 decision material for Felix: TRT-LLM's expert GEMM time vs pie's (you have it), the projected pie W4A8 win if pie's select reached TRT-LLM's ~517 TFLOP/s, and an hours estimate for what it needs (K = 2880 tail, fused GLU/bias/clamped SwiGLU, the precision-policy change). No kernel building.
3. One line on the 120b batching cell (−30% vs TRT-LLM on the site): is it the same precision cause, or the admission cap the A100 run is fixing? One measurement at the site's best point is enough. Don't chase it.
4. report.md with the playbook's sections, as they apply to a no-fix outcome. No pr.md unless item 1 turns up something. Then READY FOR REVIEW.
