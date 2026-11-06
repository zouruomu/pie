# status: 20261106-0910-pro6000-x1-gpt-oss-120b-pbatch

**12:35Z: report.md written; independent self-review running; then READY FOR REVIEW (~13:15Z). No fix, so no pr.md (note 4).**

## Verdict
Both cells red on main dc6282d7, and the gap is TRT-LLM's MoE precision on sm_120. No W4A16 gap at or above 4.5% anywhere.
- Best points: 20b pie 47.0k vs TRT-LLM 72.2k (-35%), 120b 29.1k vs 43.4k (-33%); the same -33..-35% from pf32 to pf1024.
  3 alternating rounds at pf128-1024, spreads <2%. pie leads at 120b pf4 and ties at pf8.
- TRT-LLM 1.4.0rc2.post1 runs gpt-oss's experts as W4A8 (MXFP4 x MXFP8) on sm_120, its only path there; pie runs W4A16, as the release stores.
- Equal-token nsys at pf256: the MoE is 97% of the delta on both models (120b expert GEMMs +1660 ms, glue -168; rest +49 ms, ~1% of wall).
- pie vs vLLM 0.32 (W4A16): pie leads or is level everywhere (120b pf1024 -0.8%, pf256 +0.2%, 3 alternating rounds; 20b +4.3%..+30%).
- The site's 120b 13.8k (ee9098f60) does not reproduce: ee9098f60 runs 28.7k here, with the same kernels as main. The site's 120b vLLM (22.3k)
  also ran low that day. #1430 changes nothing for gpt-oss.
- W4A8 decision material: pie's existing sm_120a W4A8 select already runs 490-514 TFLOP/s at gpt-oss-120b's shape (K padded to 3072), TRT-LLM's
  ~517. It needs the K = 2880 tail, gpt-oss's GLU/bias/clamped SwiGLU, precision plumbing, and quality evidence: ~13-19 h, projected
  120b ~38-42k. Out of scope per note 3.
- 120b batching c1024 (all requests at once): pie 8,185 vs TRT-LLM 7,119 out tok/s. The site's -30% (pie TTFT 20.9 s) looks like admission, not precision.

## Orchestrator notes
- note-1 (09:30Z), note-2 (09:42Z), note-3 (10:05Z), note-4 (11:25Z): acknowledged and followed (see results/status-history.md for the running log).

## Running
- self-review subagent over report.md and results/ (output: results/review-1.md).
