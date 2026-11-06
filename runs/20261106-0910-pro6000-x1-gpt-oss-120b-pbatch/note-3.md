# Note 3 (orchestrator, 10:05Z): answer to your question

A change to gpt-oss's default MoE precision is out of scope for this PR. pie's W4A8 selects run where the release declares e4m3 rows (the H100 profile's `moe.mxfp4_fp8_from` covers DeepSeek-V4 and MiMo, whose releases quantize rows), and gpt-oss's release is W4A16. Switching it is a quality and precision decision for In Gim and Felix, not an automation PR.

So:
1. The shippable target is any gap pie can close at its current W4A16 precision. Use the per-component split from note 2: attention, dense projections, norms, the MoE at equal precision, fire and host overhead. Also compare pie W4A16 against vLLM 0.32.0 (W4A16) over the pf sweep. Any point where vLLM leads by ≥4.5% is in scope.
2. Keep the W4A8 work as measurement only. Report pie W4A8 (if your worktree build runs) vs TRT-LLM, the quality numbers from note 2 item 2 if cheap, and an hours estimate for the K = 2880 tail. That goes in report.md as a decision for Felix. Don't spend more than ~2 h on it.
3. If nothing at W4A16 is red, write that up and hand over early. Don't stretch to the 19:10Z target.
