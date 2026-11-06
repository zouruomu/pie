# Brief: 20261106-0130-b200-x1-minimax-m3-1-flash-decode

This is a retry of run 20261105-2035-b200-x1-minimax-m3-1-flash-decode, which was abandoned before any measurement (its agent exited 6½ minutes in, and a person then stopped the pod). Its evidence branch `r2g-evidence/20261105-2035-b200-x1-minimax-m3-1-flash-decode` has only a first status.md. Start fresh.

## The cell

**B200 × MiniMax-M3.1-Flash (miniature) × decode** (`raw` workload): pie trails vLLM by 36.3%.

| engine | version | decode tok/s | prefill tok/s | TTFT p50 ms | ITL p50 / p99 ms | measured (UTC) |
|---|---|---|---|---|---|---|
| pie | `0d857cda6` | 111.17 | 15734.68 | 45.5 | 8.99 / 9.28 | 2026-11-03 20:07 |
| vLLM | 0.32.0 | **174.43** | 2846.67 | 251.87 | 5.73 / 6.86 | 2026-11-03 07:13 |
| SGLang | 0.5.22 | 128.24 | 6035.74 | 118.53 | 7.80 / 8.20 | 2026-11-02 11:58 |

- Site fields: hardware `b200-x1` (1 × NVIDIA B200, sm_100), model `minimax-m3.1-flash`, scheme `fp8` (the MXFP8 release: e4m3 planes beside block scales), gain −0.3627, pie commit `0d857cda6` (2026-11-03 19:38 PT), site stale tag `b200-full`.
- **Miniature:** the site labels it "60 of 60 layers": all 60 layers kept, `mini_k` 0.125 of the routed experts (16 of 128, top-4 routing), 124.25 B parameters (23.0 B active, 2.458 B embedding). The full model is 428 B / 224 GiB and doesn't fit one B200. Reproduce the exact cut the eval harness uses: `scripts/bench/shrink_checkpoint.py` (it has a `minimax_m3` entry and fetches only the kept tensors by HTTP range) and the harness in anton-mel/stats. Every engine must load the same cut.
- Pie's ITL is the gap: 8.99 ms against vLLM's 5.73 ms. On H100 SXM the same miniature decodes at 12.09 ms in pie and 12.48 ms in vLLM, so vLLM gains 2.2× from Hopper to Blackwell and pie only 1.34×. A first hypothesis to test, not a conclusion: vLLM uses sm_100's block-scaled MXFP8 MMA (tcgen05) for the expert and dense GEMMs at decode, and pie doesn't. Profile before believing it.
- The other B200 modes for this model are green: pbatch pie 35233 vs vLLM 11545, batching 12233 vs 6072, prefill (raw) 15735 vs 2847. A fix must not give these back.

## Workloads (site `workloads` table)

- **raw** (this cell): concurrency 1, 512-token prompt, 256 output tokens (`raw-512-256`). decode tok/s is the scored metric.
- throughput (batching): 128 in, 128 out; sweep c4, c8, c16, c32, c64, c128, c256, c512, c1024.
- prefill_batch: 512-token prompts, 1 output token; sweep pf4 … pf1024.
- Exact flags: `scripts/bench` and `.github/workflows/evals.yml`.

## Related red cells (other hardware, same model and mode; not on this pod)

- H200 × minimax-m3.1-flash decode: pie 77.86 vs vLLM 86.80 (−10.3%), pie `f21c4619d`.
- A100 SXM × minimax-m3.1-flash decode: pie 50.73 vs vLLM 57.91 (−12.4%), pie `d9c9582b6`.
- If your cause is general rather than sm_100-specific, say so in the report; don't chase these here.

## Commits on main since `0d857cda6` (checked 2026-11-06 01:20Z, main at e2e9d1e)

Nothing aimed at this cell. Possibly relevant:
- `a27d403` minimax_m3: router logits land f32 (#1410) — `models/minimax_m3.poem`, numerics of the router. The only commit since that touches the model.
- `3d6cce5` (#1433) and `1112a02` (#1434): block-FP8 selects on **Hopper** (K in whole 128s; 32-route slices). Hopper only, but read them: they show how the maintainers shape block-FP8 MoE selects.
- `e4bb499` (#1436) runtime: a piece writes only the working sets with rows in it.
- `05dace7`, `0d857cd`-era RS store changes: host bookkeeping.
- Since then only unrelated commits (`f1f7655` engine-wgpu, `75eb46b` server-web, `e2e9d1e` L4 profile).
- No commit touched `crates/engine-cuda/profiles/sm100*`. Open PRs on pie-project/pie: none on MiniMax, B200 or FP8 (#1437, `vllm_bench --max-num-seqs`, is open and touches the vLLM harness for batching; it doesn't change raw decode at concurrency 1, but note it).

## Ledger notes

- The earlier attempt on this cell measured nothing (see above). Its orchestrator asked one question you should answer early (15 minutes at most, not a blocker): the site lists this miniature at 124.25 B parameters (23.0 B active, 2.458 B embedding, `mini_k` 0.125, 60 layers). Count your cut's parameters and put both numbers in report.md. If they disagree by more than rounding, say how (e.g. a different expert count) and whether it can matter for decode at concurrency 1 (top-4 routing reads 4 experts a layer either way).
- That attempt's agent was killed 6½ minutes in, before its first real check-in, cause unknown. Run sync.sh within your first 15 minutes, with a status.md that says what you've started, so the orchestrator can tell you're alive. Use `date -u` for the UTC line (the last agent's clock line was 11 minutes off).
- Another run is active right now, not overlapping: A100 SXM × gemma-4-31b batching (it changes the runtime's KV admission cap, which doesn't bear on concurrency-1 decode).
- From earlier runs: grep the profiles and commits since the measured commit for the site's pie number before trusting it; the site metric is each engine's best point. Build the engine-cuda GPU suite at `CARGO_PROFILE_DEV_OPT_LEVEL=2` and run nothing else on the GPU while it runs.

## Time plan

- **Deadline:** 20 h from pod start (the pod terminates itself).
- **Early verdict:** about 2.5 h in (the cut download and three engine installs take time): the decode gap on current main, pie vs vLLM and SGLang, ITL with spread over ≥3 rounds, SM clocks, at the top of status.md. If it's green, check pbatch/batching curves for mid-curve red points (pf4–pf32 and c4–c16 are the likeliest), then hand over quickly if nothing is red.
- **Per-step attribution** (nsys, pie vs vLLM, component table summing to the ITL delta): by about 5 h.
- **Target hand-over:** about 10 h in, leaving the rest for review rounds.
- Check in with sync.sh at least hourly; notes arrive as `runs/<run>/note-N.md`.

## Expected at hand-over (per the PLAYBOOK)

- `report.md`: reproduced baseline on current main (commit, clocks, rounds), the per-step table, root causes with pie `file:line` and vLLM's `file:line` at 0.32.0, the fix line by line, before/after tables, full token-id equality (or quantified numerics if a kernel changes them), cross-model checks on B200 (at least one other block-FP8 model, e.g. glm-5.3 or another fp8 MoE you can cut, and one bf16 model if the change touches shared paths), the other loads (batching curve, pbatch, TTFT) for no regression, memory fit, tests (logs in `results/tests/`, GPU suite on the fix build, failures rerun on base), remaining gap broken down, side findings, playbook notes.
- `pr.md`: title in pie's style (e.g. "Decode part of the pie vs vLLM gap on B200 (…)"), one or two dense paragraphs.
- `results/`: raw JSON and logs behind every number, the engine sha256 per run, the analysis scripts.
- If the cause is a missing sm_100 kernel that would take more than this run, say so early in status.md with the measured size of the win it would buy, and hand over the attribution instead of a half-built kernel.
