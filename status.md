# status: 20261106-0910-pro6000-x1-gpt-oss-120b-pbatch

**READY FOR REVIEW (12:40Z).** No fix, so no pr.md and no fix branch push (note 4). report.md has the full write-up; results/ has the raw data.

## Verdict
Both cells red on main dc6282d7. The gap is the MoE expert GEMMs, which TRT-LLM 1.4.0rc2.post1 runs at W4A8 (MXFP4 x MXFP8) on sm_120,
its only gpt-oss MoE path there. pie runs W4A16, as the release stores. Nothing at W4A16 is red at or above 4.5%.
- Best points: 20b pie 47.0k vs TRT-LLM 72.2k (-35%); 120b 29.0k vs 43.4k (-33%). -33..-35% from pf128 up, 3 alternating rounds.
- Equal-token nsys at pf256: the MoE is 97% of the kernel-time delta on both models; everything else is ~1% of wall.
- pie vs vLLM 0.32 (W4A16 MARLIN): pie leads or is level at every pf on both models (120b pf1024 -0.8%, pf256 +0.2%).
- The site's 120b 13.8k does not reproduce: ee9098f60 runs 28.7k here, with the same kernels as main. #1430 changes nothing for gpt-oss.
- W4A8 decision material for Felix: pie's existing sm_120a W4A8 select runs 459-481 useful TFLOP/s at gpt-oss-120b's shape.
  - Projected 120b ~38.8-43.0k (vs 43.2k) and 20b ~58-61k (vs 71.5k).
  - Needs the K = 2880 tail, gpt-oss's GLU/bias/clamped SwiGLU, precision plumbing, and quality evidence: ~15-21 h.
- 120b batching c1024 (n=1, all requests at once): pie 8,185 vs TRT-LLM 7,119 out tok/s. The site's -30% is unexplained here;
  pie's 20.9 s site TTFT points at admission (hypothesis).
- Self-review done (results/review-1.md); its findings are addressed in report.md.

## Orchestrator notes
- note-1 .. note-4: acknowledged and followed (running log: results/status-history.md).
