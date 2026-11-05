# Outcome: 20261105-2033-h100-sxm-x1-gpt-oss-120b-batching

**Abandoned at 1 h 4 min.** The pod (`lthznv0blokitl`, H100 80GB HBM3, US-NE-1, $3.99/h) was stopped "by user" at 21:38:06Z. The orchestrator didn't do it, and neither did the bootstrap, which deletes rather than stops. The parallel run 20261105-2035-b200-x1-minimax-m3-1-flash-decode's pod was stopped at 21:38:05Z, a second earlier. Teammates resumed the mi300x-eval-1..4 pods at 21:36–21:37Z. A pod stopped on the shared account by a person was not restarted. Container disk doesn't survive a stop, so the pushed evidence is all there is. Cost: about $4.25.

## What the evidence holds (round 1 of 3, main e4bb499c, verified from `results/r1-*.json`)

Batching: 128-token prompts as identical HF-rendered ids, 128 out, max(256, 2c) requests, bench defaults without `--report-timing`.

| c | pie | vLLM 0.32.0 | pie vs vLLM |
|---|---|---|---|
| 32 | 1933 | 1989 | −2.8% |
| 64 | 3112 | 3036 | +2.5% |
| 128 | 5031 | 3818 | +31.8% |
| 256 | 7437 | 4487 | +65.7% |
| 512 | 6511 | 5197 | +25.3% |
| 1024 | 5795 | 5030 | +15.2% |

- **Best point against best point:** pie 7437 vs vLLM 5197. The site has pie 4560 vs vLLM 7198. This is one round with no spread, and the setup doesn't yet match the site's harness flags.
- **Open question:** whether the site's vLLM number (38% above the pod's) comes from different harness flags. `scripts/bench/vllm_bench.py` sets `max_num_seqs = num_requests` on one path (see PR #1437), and the site's vLLM TTFT p50 of 14.6 s points to a deep queue. Note 1 asked the pod to replicate the site's exact invocation; it never read it.
- **KV pool:** pie 5.5–6.5 GiB, varying with concurrency; vLLM 7.68 GiB (111k tokens).
- **Leads the pod named but didn't measure:**
  - gpt_oss.poem pages its sliding-window layers in the full KV space, where Gemma 4 gives them a windowed ring space;
  - a 2047 MiB graph-body reserve of which 139 MiB is used;
  - #1368's admission cap.
- **SGLang 0.5.22:** OOM at load while swizzling MXFP4 on 80 GB.
- **Not reached:** raw prefill and pbatch.
