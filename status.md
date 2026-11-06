# Status: 20261106-0130-b200-x1-minimax-m3-1-flash-decode

UTC: 2026-11-06 08:34Z
Phase: **READY FOR REVIEW**. Fix branch r2g/20261106-0130-b200-x1-minimax-m3-1-flash-decode at d06613bc4 (one commit, rebased onto upstream main dc6282d76; engine byte-identical to the tested 3da03d2c1). report.md, pr.md, identify-bug.md and results/ are complete.

## Verdict
- The gap is real on B200, but no released vLLM (0.32.0, 0.33.0) or SGLang (0.5.22) loads MiniMax-M3.1-Flash's NVFP4 experts, so the site's competitor numbers can't be this checkpoint. Same-weights comparison on a 60-layer, 16-expert cut of MiniMax-M3-MXFP8: pie trails vLLM 0.32.0 by 28%, SGLang 0.5.22 by 22%.
- Fix: MiniMax's decode selected attention over a cluster of up to 8 blocks a kv head (it ran a block a kv head). c1, 3 rounds, SM 1965 MHz: M3.1-Flash 8.976 -> 8.620 ms (+4.1%), M3-MXFP8 9.623 -> 9.286 ms (+3.6%; vs vLLM 28.0% -> 25.4% behind). c2 +1.8% / +2.5%, c4 -0.1% / +1.7%, c8-c64 and pf4-pf32 unchanged within spread; Qwen3.8-Flash-Next mini +3.4%; Qwen3-0.6B unchanged, identical tokens.
- Numerics: M3-MXFP8 tokens identical 8/8 x 256; M3.1-Flash diverges at bf16 near-ties; teacher-forced decode KL below the scalar selected walk's on all 8 probed requests.
- Tests: GPU suite on the final commit: engine-cuda 69/69 lib, 58/59 integration, 5/6 EP; kernels-cuda 34/35 lib, 142/144 integration. All five failures reproduce on main e2e9d1e (logs in results/tests/base-*.log). New MiniMax-shaped cluster test; an unweighted fold fails it.
- Not shipped, handed over as attribution: q/k norm + RoPE + KV append (two 1-CTA epilogue slices, ~0.7 ms a step; a reverted prototype of the retired hand rule buys ~4%, the shippable form is skeleton/compiler work), the expert glue, the W4A4 expert GEMM on M3.1-Flash.

## Notes and review
- Notes 1-5 acknowledged and acted on (see report.md). Independent review (/root/r2g/review/review-1.md): both blockers (fmt, test coverage) fixed, the cluster rule tightened after the first build's c8 loss, every number recomputed.
- Pod state worth knowing: ~/.pie/models/minimax_m3.poem carries a local edit (the mini's [MX8, nvfp4a4] and MX8 deployments ahead of bf16) so pie serves the release precision; see identify-bug.md. Worktree /root/wt-base holds main e2e9d1e for base reruns.
