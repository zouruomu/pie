# Status: 20261106-0130-b200-x1-minimax-m3-1-flash-decode

UTC: 2026-11-06 01:31Z
Phase: setup, nearly done
Done: pie engine built at e2e9d1e (sha256 b0ca94bb…), bench inferlet built, vLLM 0.32.0 and SGLang 0.5.22 installed in their own venvs.
Running: the cut (MiniMaxAI/MiniMax-M3.1-Flash, layers 0-59, 16 of 128 experts, 43.41 GiB, MXFP8 projections + NVFP4 experts; same command as the last attempt), ~40/43 GiB.
Next: 512-token prompts, then pie / vLLM / SGLang raw rounds (3+ alternating), clocks recorded.
Early leads (unmeasured): the B200 profile already runs NVFP4 experts W4A4 on tcgen05 (moe.w4a4_fused_from = 1), so "no block-scaled MMA" is doubtful; H200's profile has sdpa.decode.gqa_min_group = 2 (MiniMax raw decode 77.9 -> 93.9 there) and B200's doesn't.
Blocker: none.
ETA early verdict: ~03:00Z. Hand-over target: ~11:15Z.
