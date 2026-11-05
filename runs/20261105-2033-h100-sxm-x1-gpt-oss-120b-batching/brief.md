# Brief: 20261105-2033-h100-sxm-x1-gpt-oss-120b-batching

Pod: 1 × NVIDIA H100 80GB HBM3 (site hardware `h100-sxm-x1`, "NVIDIA H100 SXM", sm_90, profile `crates/engine-cuda/profiles/sm90/nvidia-h100-80gb-hbm3.toml` over `sm90.toml`).

## Why this target

gpt-oss-120b (full checkpoint, MXFP4, not a miniature) is red on most datacenter cards in the site's newest release, while gpt-oss-20b is green on the same cards. On H100 SXM: batching −36.6%, raw prefill −24.4%, pbatch −16.6%. On H200 (same sm_90): batching −22.6%, prefill −22.9%, pbatch −27.1%. Also batching on B200 −17.8%, RTX PRO 6000 −27.2%, A100-SXM −18.0%. So the cause is most likely specific to the 120B checkpoint (128 experts, 36 layers, ~61 GiB of weights), not to gpt-oss in general. One cause that explains several cards is worth much more than one cell.

## The cells (site fields, release `mac-batched`, db updated 2026-11-05 19:41Z)

All three: model `gpt-oss-120b`, scheme `mxfp4`, not a miniature, pie commit `ee9098f60`, site tag `h100-fill`. Competitors: vLLM 0.32.0, SGLang 0.5.22.

| mode | workload | pie | best | best engine | gain | pie measured | best measured |
|---|---|---|---|---|---|---|---|
| batching | throughput (output tok/s) | 4560.73 | 7197.59 | vllm 0.32.0 | −36.6% | 2026-11-03 15:17 | 2026-11-03 15:25 |
| prefill | raw (prefill tok/s) | 10347.02 | 13683.67 | sglang 0.5.22 | −24.4% | 2026-11-03 12:39 | 2026-11-03 14:22 |
| pbatch | prefill_batch (prefill tok/s) | 31661.8 | 37967.56 | sglang 0.5.22 | −16.6% | 2026-11-03 15:29 | 2026-11-03 15:31 |

Other site fields at each engine's best point (useful hints, not facts):
- **batching:** pie ITL p50 30.5 ms, TTFT p50 1.4 s; vLLM ITL p50 29.1 ms, TTFT p50 14.6 s; SGLang 3797 tok/s. Similar ITL but 1.6× the throughput suggests vLLM's best point runs many more lanes a step (a high concurrency, deep queue). Check whether pie seats fewer sequences: the KV pool left beside 61 GiB of weights, how sliding-window layers (half of gpt-oss's) are paged, the admission cap, and pie's own c512/c1024 points.
- **raw:** pie decode 276.8 tok/s (green, +29% vs vLLM 214.5), prefill 10347 vs SGLang 13684 / vLLM 12760.
- **pbatch:** SGLang 37968, vLLM 28802, pie 31662. TTFT p50 pie 1.09 s, SGLang 2.02 s.
- **H200 same model (related, not claimed):** batching pie 9971 vs SGLang 12883; prefill 10214 vs 13247; pbatch 27173 vs SGLang 37267 (pie lower on H200 than H100 here, worth a look). All at `ee9098f60`.

Note the H100 profile's MXFP4 prefill path: `moe.mxfp4_bf16_from = 1024` keeps an exact bf16 copy of gate-up banks for big fires, and was tuned on gpt-oss-20b. A bf16 copy of 120B's experts can't fit 80 GB, and #1401 (below) now skips copies that don't fit, so 120B's prefill likely runs the register-dequant select (~330 TFLOP/s per that comment). Verify; this is a hypothesis for prefill/pbatch.

## Workloads (database `workloads` table)

- **Batching (`throughput`):** concurrency 32 nominal, 128-token prompts, 128 output tokens; sweep c4, c8, c16, c32, c64, c128, c256, c512, c1024. The site scores each engine's best point.
- **Raw:** 1 request, 512-token prompt, 256 output tokens (`raw-512-256`).
- **Prefill batch (`prefill_batch`):** 512-token prompts, 1 output token; sweep pf4, pf8, pf16, pf32, pf64, pf128, pf256, pf512, pf1024.

Read `scripts/bench` and `.github/workflows/evals.yml` for the exact flags.

## What changed on main since `ee9098f60` (main is `e4bb499c0` at claim time)

No change to `profiles/sm90*` or `models/gpt_oss.poem`. Possibly relevant:
- `c57e793` #1368 KV admission: seated-first pools cap admission at five sequences a declared-context sequence holds. **The H100 profile has `seated_first = true`**, so this cap now applies to gpt-oss-120b's small pool. The previous run measured it costing L4 c256 13%. Measure its effect here.
- `882f246` #1401 MXFP4 bf16 copies only where they fit the free memory (`crates/engine-cuda/src/weights.rs`).
- `1716873` #1400 arena: attention score planes held out of the placement (memory layout).
- `0ea459a` #1418, `0dc19aa` #1420, `0d857cd` #1422, `05dace7` #1423, `e4bb499` #1436: runtime host-side bookkeeping (KV store lock, RS store, working sets).
- `1c0ae1f` #1405, `3d6cce5` #1433, `1112a02` #1434: Hopper bf16 / block-FP8 MoE selects (not MXFP4, but same kernels family).

So the site's numbers may be off for batching. Re-measure everything on current main before attributing.

Nobody else's open PR on pie-project/pie touches this (checked 2026-11-05 20:30Z: #1438 wgpu, #1437 vllm_bench `--max-num-seqs`, #1435 cerebras, #1432 ours on L4). Note #1437 changes vLLM's bench flag set; read it to know how the site will drive vLLM.

## Ledger notes from earlier runs

- gpt-oss-20b on L4: `moe.mxfp4_mma_per_from` (select by route vs run kernel) mattered at 1–2 routes an expert. gpt-oss-120b has 128 experts and top-4, so at c32 decode it has ~1 route an expert and at c256 ~8: check which MXFP4 select variant 120B's decode steps take on sm_90 and whether it's the right one. #1355 did the same for the RTX 4090.
- Expect the site metric to be each engine's best point of its sweep. Measure the whole curve.
- engine-cuda GPU suite: dev build at `CARGO_PROFILE_DEV_OPT_LEVEL=2`, `--test-threads=1`, nothing else on the GPU while it runs.
- `linear::moe_select::the_routed_selects_answer_the_routed_dot` fails off sm_120a/121a at #1274's first W4A8 case: expected, not yours (verify on base anyway).

## Scope

Primary: batching. Also prefill and pbatch, in that order of priority, if the attribution points at a fix of reasonable size. If they need separate fixes, say so; they may ship as separate PRs. If gpt-oss-20b on this card is affected by your change, measure it (it's green on the site: decode +26%, batching −2%, pbatch −4%).

Downloads: `openai/gpt-oss-120b` (~61 GiB). Container disk is 300 GB.

## Time plan

- **Deadline:** 20 h from pod start (the pod terminates itself).
- **Early verdict, about 2.5 h in** (the download and builds are big): the whole batching curve (pie vs vLLM 0.32.0 and SGLang 0.5.22, at least c32–c1024), raw prefill and pf32–pf256, on current main. Put the gap per point and its spread at the top of status.md, plus pie's KV pool size and seats against vLLM's (its log prints the KV cache size and max concurrency).
- **Target hand-over:** about 10 h in. The rest is for review rounds.

## Expected at hand-over (PLAYBOOK §7)

report.md, pr.md and results/ as in the PLAYBOOK: the reproduced curves on current main, a per-step attribution that sums to the measured gap, causes with `file:line` in pie and in the competitor, the fix line by line, base/fix/competitor A/B (≥3 alternating rounds, engine sha256 in each JSON), full token-id comparison next to base-vs-base, cross-model checks on this card (gpt-oss-20b at least, plus any other model the change touches), memory fit, tests (relevant unit tests on both builds, engine GPU suite on the fix build), a final A/B after rebasing on newest main, and the remaining gap broken down.
