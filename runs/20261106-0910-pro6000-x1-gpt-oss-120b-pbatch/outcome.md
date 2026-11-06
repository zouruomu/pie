# Outcome: 20261106-0910-pro6000-x1-gpt-oss-120b-pbatch

**Verdict: no fix, no PR.** Both cells are red on main dc6282d76, and the whole gap is a precision difference: TensorRT-LLM 1.4.0rc2.post1 runs gpt-oss's MoE experts at W4A8 (MXFP4 × MXFP8) on sm_120, which is its only path there. pie runs W4A16, as the release stores the model. At equal precision nothing is red: pie leads or ties vLLM 0.32 (MARLIN W4A16) at every pf point.

## Numbers (verified by the orchestrator from results/ raw JSON)
- 20b, best points: pie 47.0k vs TRT-LLM 72.2k tok/s (−35%). 120b: 29.0k vs 43.4k (−33%). The gap is −33…−35% from pf128 to pf1024 over 3 alternating rounds (spreads ≤2%), and −29…−31% at pf32. At pf4/pf8, pie leads (120b) or trails by 9–10% (20b).
- pie vs vLLM: 20b +4.3%…+30% at every point; 120b pf256 +0.2%, pf1024 −0.8% (pie n=5, vLLM n=3).
- pf256 at equal tokens (131,072), nsys kernel time: 120b pie 4582 vs TRT-LLM 3041 ms; expert GEMMs 3478 vs 1818. 20b: 2822 vs 1790; expert GEMMs 2091 vs 991. The MoE is 97% of the delta, and everything else about 1% of wall. TRT-LLM's MoE kernel is the CUTLASS SM120 block-scaled e2m1 × e4m3 grouped GEMM.
- Engine sha256: main 22af6e94…b143 on all 48 pie baseline runs; ee9098f60 4f26304c…; ea800efeb 7539275920b1….
- TRT-LLM with KV block reuse off reproduces the site (20b 71.5k vs 71.3k; 120b 43.2k vs 42.4k).
- The site's 120b pie 13.8k (ee9098f60) doesn't reproduce: that commit runs 28.7k here. The site's vLLM 120b row also ran 24% low that day.
- Review 1: ea800efeb vs main, 20b, 3 alternating rounds, both at 448 prompt tokens: main +1.7% (pf32), +1.2% (pf256), −0.1% (pf1024). No regression. ea800efeb doesn't reach the site's 49.8k either.

## Decision material for Felix (not done; out of scope per note 3)
A W4A8 gpt-oss deployment on sm_120. pie's existing `moe_select_mxfp4_w4a8_ws` already runs 490–514 TFLOP/s at the 120b shape with K padded to 3072; TRT-LLM's GEMMs run ~517. It needs:
- the K = 2880 tail;
- gpt-oss's bias and clamped SwiGLU in the epilogue;
- loader plumbing (`mxfp4a8g32` over the stock checkpoint);
- quality evidence.

Estimate ~15–21 h, projected 120b ~39–43k and 20b ~58–61k. It changes gpt-oss's numerics on prefill and wide decode fires.

## Side findings
- pie's harmony template drops the 64-token system preamble HF renders (`crates/chat-template/src/harmony.rs:142-147`), so pie prefills 448 tokens where the others prefill 512. Throughput per token is unchanged at equal counts.
- 120b batching at c1024 (one round): pie 8,185 vs TRT-LLM 7,119 out tok/s. The site's −30% (pie TTFT 20.9 s) points to admission (#1368's cap), not precision.
- RTX 5090 and PRO 4500 very likely share the cause (same W4A16 staged select; TRT-LLM W4A8 on all sm_120). Not measured.

## Run
- Pod qqnk9anspw173j, US-PA-1, $2.49/h, 250 GB disk. Created 09:08Z; early verdict 10:25Z; handed over 12:40Z; review 1 answered 13:09Z; closed ~13:20Z. Deadline was 05:08Z.
- About 4.2 h of pod time, ~$11.
- 4 notes and 1 review round.
- Evidence: branch r2g-evidence/20261106-0910-pro6000-x1-gpt-oss-120b-pbatch (report.md, results/).
