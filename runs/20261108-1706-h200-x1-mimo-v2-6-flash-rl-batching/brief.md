# Brief: 20261108-1706-h200-x1-mimo-v2-6-flash-rl-batching

First run on this model. No earlier evidence branch to build on.

**Before anything else:** run sync.sh within your first 10 minutes with a status.md that says you started. The last four pods' agents died at start on a revoked Claude token, so the orchestrator is watching for your first check-in to know the token works.

## The cells (H200 SXM, 1 GPU, sm_90; all measured 2026-11-08 at pie `bb2d6bf43`)

**1. Batching (throughput workload), primary: pie trails vLLM by 50.5%.** Site best points, all at c1024:

| engine | version | output tok/s | ITL p50 / p99 ms | TTFT p50 ms | lanes ≈ out × ITL |
|---|---|---|---|---|---|
| pie | `bb2d6bf43` | 2295.75 | 60.41 / 393.05 | 93381 | ~140 |
| vLLM | 0.32.0 | **4635.33** | 142.08 | 84416 | ~660 |
| SGLang | 0.5.22 | 3637.14 | 87.94 | 124988 | ~320 |
| TRT-LLM | — | unsupported | | | |

- Site fields: `h200-x1`, model `mimo-v2.6-flash-rl` (XiaomiMiMo/MiMo-V2.6-Flash-RL, 309 B / 15 B active, 48 layers, 256 experts top-8), scheme `mxfp4`, gain −0.5047, pie measured 2026-11-08 04:09, best measured (vLLM) 2026-11-08 03:48, site tag `h200-rerun`. pie_stale: tag `h200-rerun`.
- **Miniature:** "48 of 48 layers", `mini_k` 0.062 (16 of 256 experts), site counts 157.258 B params, 15.0 B active, 1.25 B embedding. This is the first full-depth measurement; the 0.5.3 tag measured a 6-layer cut where pie led vLLM 3× (32214 vs 11094 at c256). Reproduce the eval harness's exact cut: `scripts/bench/shrink_checkpoint.py` has a `mimo_v2` entry (its default is layers 0-5; the site's cut keeps all 48 layers and 16 experts, so find the flags the harness used in `scripts/bench`, `.github/workflows/evals.yml` and anton-mel/stats). Every engine loads the same cut. Count your cut's parameters and put both counts in report.md.
- **What the numbers say (a reading, not a conclusion):** pie's per-lane decode is fast (ITL 60 ms vs vLLM's 142 ms) but it decodes ~140 lanes against vLLM's ~660, and its TTFT p50 is 93 s, so requests are queueing. Its single-stream decode leads too (raw ITL 6.2 vs 6.98 ms). So look first at how many sequences pie seats: KV pool size, page accounting, admission (#1368's cap in `crates/runtime/src/bootstrap.rs`, #1488's memory governor), and the load fit (`crates/engine-cuda/src/store.rs`).
  - Things to check in that order of suspicion: `models/mimo_v2.poem` pads every head to 256 (`PADDED`, keys 192 and values 128 wide) and declares the kv planes `[plane, plane]` with `live = [192, 128]`: are KV pages stored 256+256 wide, 1.6× the live bytes? The 39 windowed layers get `c.kv_space(m.kv, m.window)` (window 128): does the admission/fit count them as windowed, or charge them a full context? The admission cap's arithmetic for a model with two kv spaces. vLLM's hybrid allocator for SWA layers. Measure each before you believe it.
- Score the whole curve (c4 … c1024), not only c1024: at every point compare pie with the best competitor at that same point.

**2. Prefill (raw workload), secondary: pie trails vLLM by 30.5%.**

| engine | version | prefill tok/s | TTFT p50 ms | decode tok/s | ITL p50 ms |
|---|---|---|---|---|---|
| pie | `bb2d6bf43` | 4897.38 | 112.87 | 161.18 | 6.20 |
| vLLM | 0.32.0 | **7043.56** | 78.01 | 143.32 | 6.98 |
| SGLang | 0.5.22 | 5835.10 | 94.34 | 113.69 | 8.80 |

- pie measured 2026-11-07 22:10 at `bb2d6bf43`. A 512-token prompt at concurrency 1. A first guess to test: the 256-wide padded heads cost the prefill attention (and qkv split) about 1.6× the work. Do this only after the batching cell is attributed, or in parallel if the GPU is otherwise idle.
- pbatch on this cell is green (pie 12466.61 at `f21c4619d` vs vLLM 12779.70, −2.4%, pf16-512); re-measure it as a regression check.

## Workloads (site `workloads` table)

- **throughput (batching):** c4: 128 in / 256 out; c8: 128/128; c16: 128/256; c32: 128/128; c64, c128, c256, c512, c1024: 128/256. Scored metric: output tok/s, each engine's best point.
- **raw (prefill and decode):** concurrency 1, 512-token prompt, 256 output tokens (`raw-512-256`).
- **prefill_batch:** 512-token prompts, 1 output token, pf4 … pf1024.
- Exact flags: `scripts/bench/{pie,vllm,sglang}_bench.py` and `.github/workflows/evals.yml`. Match the site's harness flags before attributing (an earlier run lost an hour comparing against vLLM run with different `max_num_seqs`; #1437 added `--max-num-seqs` to vllm_bench).

## Related red cells (not claimed; same model)

- RTX 4090 × mimo (6-layer mini): batching pie 20816 vs SGLang 22571 (−7.8%), pbatch 51501 vs 54303 (−5.2%). Lanes there are equal (~180 each), so it is probably not the same cause.
- PRO 6000 ×4 prefill −9.9% and H100 ×4 decode −6.3%: old (0.5.3, October) measurements.
- If your cause is general (any model with padded heads, or with windowed kv spaces: gpt-oss, Gemma 4, Muse), say which other cells it should move, and check one other model on this H200 if the change touches shared code. gpt-oss-120b batching on this card (pie 9688 vs SGLang 12529) is red too, but there pie seats more lanes than SGLang at a slower ITL, so don't assume it shares this cause.

## Commits on main since `bb2d6bf43` (checked 2026-11-08 17:10Z, main at `9d134db`)

Nothing aimed at MiMo, sm_90 or SWA KV. Touching the engine:
- `276ebdd` (#1490) engine-cuda: keep the state slots a deployment states beside the kv pages (`serve/load.rs`, `store.rs`, +37). For hybrid recurrent trunks; MiMo has no state slots, but it edits the load fit you may touch, so read it.
- `4a6d495` (#1493) Qwen3.6 decode on a 100 KiB SM (GDN kernel; not sm_90).
- The rest is engine-xla, engine-metal and kernels-xla. No commit touched `crates/engine-cuda/profiles/sm90*` or `models/mimo_v2.poem`.
- Open PRs on pie-project/pie: none on CUDA, MiMo or admission (#1495/#1496 are kernels-xla).
- Felix's own #1485 (B200 MiniMax selected attention) is in `bb2d6bf`; unrelated.

## Ledger notes that apply

- An A100 Gemma-4-31B batching run (20261105-2035) found pie seating fewer lanes than vLLM because of #1368's admission cap (`bootstrap.rs:318-340`) and the pool fit's coarse map unit (`Pools::fit_the_card`). Its fix (uncapped admission) was not shipped: it reopened over-admission on Gemma-4-26B (lanes paused to host, ITL p99 up to 680 ms). If you land on the admission cap, read that run's outcome (`runs/20261105-2035-a100-sxm-x1-gemma-4-31b-batching/outcome.md` on r2g-control) first and don't repeat it: a fix here must keep p99 ITL and pause counts at the base's level on Gemma-4-26B or another model the cap protects.
- An H100 gpt-oss-120b run noted that `gpt_oss.poem` pages sliding-window layers in the full KV space while Gemma 4 uses a windowed ring. MiMo declares a windowed space; check what it actually allocates.
- SGLang 0.5.22 OOMs loading big MXFP4 checkpoints on 80 GB; the H200 has 141 GB. If it fails, try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Grep the profiles and commits since the measured commit for the site's pie number before trusting it; the site metric is each engine's best point.
- engine-cuda GPU suite: build at `CARGO_PROFILE_DEV_OPT_LEVEL=2` and run nothing else on the GPU while it runs. `the_routed_selects_answer_the_routed_dot` fails off sm_120a/121a at its first W4A8 case on base too; don't attribute it to your change.
- Notes from the orchestrator reach you through sync.sh's output. Don't call sync.sh from background queue scripts: it marks notes as seen when it prints them, and notes printed into a queue log are lost.

## Time plan

- **Deadline:** 18 h from pod start (the pod terminates itself).
- **First check-in:** within 10 minutes (see the top).
- **Early verdict:** about 2.5 h in (the cut download and three engine installs take time): the batching curve on current main, pie vs vLLM and SGLang, ≥3 rounds at the points that decide it (c256, c512, c1024 at least), with the lanes each engine actually decodes and pie's KV pool size, at the top of status.md. If it's already green everywhere on the curve, check prefill, then hand over quickly.
- **Attribution** (where the missing lanes go, in GiB and pages, pie vs vLLM): by about 5 h.
- **Target hand-over:** about 9 h in, leaving the rest for review rounds.
- Check in with sync.sh at least hourly.

## Expected at hand-over (per the PLAYBOOK)

- `report.md`: reproduced baseline on current main (commit, clocks, rounds, harness flags), the lane and memory breakdown, root causes with pie `file:line` and vLLM's `file:line` at 0.32.0, the fix line by line, before/after tables over the whole batching curve, full token-id equality base vs fix (or quantified numerics if attention math changes), the other loads (raw prefill/decode, pbatch, TTFT, ITL p99, pause/preemption counts), memory fit, cross-model checks on this H200 for any shared-code change (at least one other windowed-attention model, e.g. gpt-oss-20b or a Gemma 4), tests (logs in `results/tests/`, GPU suite on the fix build, any failure rerun on base), remaining gap broken down, side findings, playbook notes.
- `pr.md`: title in pie's style (e.g. "Decode part of the pie vs vLLM gap on H200 (…)"), one or two dense paragraphs.
- `results/`: raw JSON and logs behind every number, the engine sha256 per run, the analysis scripts.
- Rebase on upstream main before the final A/B, and rerun the final A/B after the rebase.
