# Brief: 20261106-1706-h100-sxm-x1-qwen3-6-27b-batching

## The cells (H100 SXM 80 GB HBM3, sm_90; site hardware `h100-sxm-x1`)

Qwen3.6-27B and Qwen3.8-27B, scheme `bf16`, full checkpoints (not miniatures): `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.8-27B`. The site files both as family `qwen3_6`, "dense", 27B; pie's H100 profile treats them as hybrid trunks (recurrent-state slots, see below). Best competitor is TensorRT-LLM 1.4.0rc2.post1, whose rows were all measured on 2026-11-05 (site group `a40-trtllm`), two to three days after pie's.

**Qwen3.8-27B × batching: pie −31.1% vs TRT-LLM** (pie commit `225b6b94d`, measured 2026-11-03 06:03, stale tag `h100-fill`)

| engine | version | out tok/s (best c point) | ITL p50 ms | TTFT p50 / p99 ms | latency p50 ms | measured |
|---|---|---|---|---|---|---|
| pie | `225b6b94d` | 2311.30 | 29.54 | 18737 / 27681 | 26882 | 2026-11-03 06:03 |
| vLLM | 0.32.0 | 2350.63 | 31.80 | 47964 / 103746 | 56948 | 2026-11-03 14:30 |
| SGLang | 0.5.22 | 1704.49 | 26.08 | 73452 / 146540 | 80113 | 2026-11-02 11:07 |
| TRT-LLM | 1.4.0rc2.post1 | **3356.51** | 30.46 | 1415 / 1925 | 9665 | 2026-11-05 21:01 |

**Qwen3.6-27B × batching: pie −26.1% vs TRT-LLM** (pie commit `ee9098f60`, measured 2026-11-03 14:54)

| engine | version | out tok/s | ITL p50 ms | TTFT p50 / p99 ms | latency p50 ms | measured |
|---|---|---|---|---|---|---|
| pie | `ee9098f60` | 2489.15 | 28.87 | 1324 / 10220 | 8927 | 2026-11-03 14:54 |
| vLLM | 0.32.0 | 2417.75 | 31.12 | 19859 / 47028 | 28671 | 2026-11-03 15:04 |
| SGLang | 0.5.22 | 1644.56 | 26.03 | 37353 / 73311 | 44031 | 2026-11-03 14:59 |
| TRT-LLM | 1.4.0rc2.post1 | **3366.33** | 30.47 | 1221 / 1791 | 9650 | 2026-11-05 21:55 |

Also claimed, cheap on the same pod (pbatch, TRT-LLM best):
- Qwen3.6-27B × pbatch: pie 11177.40 (`ee9098f60`) vs TRT-LLM 12198.94 (−8.4%); vLLM 11960, SGLang 11923.
- Qwen3.8-27B × pbatch: pie 11420.59 (`b2ec7e880`) vs TRT-LLM 12190.74 (−6.3%); vLLM 11413, SGLang 11371.

What the numbers suggest (hypotheses to test, not conclusions):
- **pie ≈ vLLM, TRT-LLM is the outlier.** At the same ~30 ms ITL TRT-LLM delivers ~35% more output tokens, so it either decodes ~35% more lanes per step at the same step time, or loses less time to prefill interleaving. pie's TTFT p99 (10–28 s) and the 18.7 s TTFT p50 on Qwen3.8 at `225b6b94d` say pie queues requests TRT-LLM serves at once: look at admission and seating first (state slots, kv pool size, working set, graph-body reserve), then at the step time at large batch.
- **The pattern follows card memory.** Red where 54 GB of bf16 weights leave little room: H100 80 GB, A100 80 GB (−27/−28%), L40S and A40 (minis, −17…−25%). Green or small on H200 141 GB (Qwen3.6 pie 4275 > TRT 3995; Qwen3.8 −10.3%), B200 (TRT-LLM behind pie), PRO 6000 96 GB (tie). That points at how many lanes the leftover memory seats in pie vs TRT-LLM (TRT-LLM's default `free_gpu_memory_fraction` is 0.9, and it has a separate mamba/recurrent-state cache).
- **Check TRT-LLM's numbers are fair before chasing them** (the PRO 6000 run found TRT-LLM's KV block reuse on by default; the site's row reproduced with reuse off there). Confirm: KV block reuse off (the 128-token prompts share a chat-template prefix), the same prompts and exactly 128 output tokens per request (ignore_eos, no early stop), bf16 weights and bf16 KV cache (not FP8 KV), its best c point. Find the TRT-LLM harness the site used (anton-mel/stats on GitHub). If TRT-LLM's lead is a harness artifact, say so early and move to the curve.
- **Key pie code:** `crates/runtime/src/bootstrap.rs:255-342` (admission: for a hybrid, the cap is the recurrent-state slot count `rs_seat_cap`, and the five-sequence kv cap does not apply; with fewer slots than lanes at the dispatch depth it posts one frame a lane); `crates/engine-cuda/src/serve/load.rs:110-118` (`state_slots_per_lane` × max_lanes, default `DEFAULT_STATE_SLOTS` = 256); H100 profile `crates/engine-cuda/profiles/sm90/nvidia-h100-80gb-hbm3.toml` `[engine]`: `working_set_percent = 90`, `seated_first`, `seat_to_end`, `state_slots_per_lane = 2` (its comment: "Qwen3.8-27B c256 2.29k -> 2.32k"), and `[compiler] branches`, `dag_streams_until = 256` ("Qwen3.8-27B c128 2.22k -> 2.48k").

## Workloads (site `workloads` table)

- **throughput (batching, these cells):** 128-token prompts, 128 output tokens; sweep c4, c8, c16, c32, c64, c128, c256, c512, c1024 (c32 default). Output tok/s is scored; the site takes each engine's best point. Mid-curve points where pie trails the best competitor at the same point by ≥4.5% are in scope.
- **prefill_batch (pbatch):** 512-token prompts, 1 output token; pf4 … pf1024 (pf256 default).
- raw: 1 request, 512 in, 256 out.
- Exact flags: `scripts/bench` and `.github/workflows/evals.yml`; `a654750` (#1437) added `vllm_bench --max-num-seqs`.

## Related red cells (same models, batching; not on this pod)

Say in the report whether your cause applies to each:
- A100 SXM: Qwen3.6-27B −27.2% (pie 1143.30 vs TRT 1569.67, pie ITL 46.5 vs TRT 61.9 ms), Qwen3.8-27B −27.9% (1134.02 vs 1572.87; pie TTFT p50 16.4 s).
- L40S (minis): Qwen3.6 −23.5%, Qwen3.8 −25.0%. A40 (minis): Qwen3.6 −17.1%, Qwen3.8 −21.3%.
- H200: Qwen3.8-27B −10.3% (4380 vs 4882). RTX 5090 (mini): Qwen3.8 −16.5%. RTX 4090 (mini): Qwen3.8 −6.1%. B200: Qwen3.6 −8.1% vs vLLM.
- Also red on this card but probably another cause: Gemma-4-31B batching −24.4% vs vLLM (`b2ec7e880`); the A100 run tied that model's gap to #1368's five-sequence cap (ledger 2026-11-06). Don't take it on unless your fix covers it for free.

## Commits on main since the measured commits (checked 2026-11-06 17:05Z, main at dc6282d76)

Nothing names Qwen3.6/3.8 or the H100 profile since `225b6b94d` (the H100 profile last changed in #1394, before both measured commits). Read these first, since they could change the cells:
- `0dc19aa` (#1420), `0d857cd` (#1422), `05dace7` (#1423): RS (recurrent-state) store bookkeeping: a slot release finds its recycle epoch without walking every fire in flight; flat vector for the newest toucher; recycle only while an unsettled fire touches it. All landed after both measured commits and touch exactly the hybrid state slots these models use at high concurrency.
- `e4bb499` runtime: a piece writes only the working sets with rows in it (#1436).
- `0ea459a` prompt keys hashed outside the KV store lock (#1418); `c71ff38` coalesce waits on the runtime's clock (#1408).
- `882f246` MXFP4 bf16 copies only where they fit (#1401): not relevant to bf16 dense, listed for completeness.
- Open PRs on pie-project/pie: none on Qwen, GDN, H100 or admission (#1484 is TPU GDN, #1485 is B200 MiniMax).

## Ledger and other runs

- No earlier run on these cells or models on CUDA. The A100 Gemma-4-31B run (ledger 2026-11-06) found #1368's admission cap plus a stranded pool fit (`Pools::fit_the_card`, `crates/engine-cuda/src/store.rs`, coarse map unit at working_set_percent 50); its uncapped fix over-admitted Gemma-4-26B c512 (ITL p99 430–680 ms vs main 177–360). If you change admission, measure the tail (ITL p99, pause/suspend log events) over 5+ runs at c512/c1024, not just throughput.
- The PRO 6000 run's playbook notes for TRT-LLM 1.4 (evidence branch `r2g-evidence/20261106-0910-pro6000-x1-gpt-oss-120b-pbatch`, report.md "Playbook notes"): needs `libopenmpi-dev`; load from the local HF snapshot path; nsys needs `--trace-fork-before-exec=true` and an NVTX window; engines at other commits build in ~2 min in a worktree with their own `CARGO_TARGET_DIR`; zsh runs `&` jobs at nice 5; take status times from `date -u`.
- Notes: run `/root/r2g/sync.sh` yourself in the foreground at each check-in. Notes printed by sync.sh inside background scripts are marked seen and lost (A100 run).
- engine-cuda GPU suite: build at `CARGO_PROFILE_DEV_OPT_LEVEL=2`, run single-threaded, run nothing else on the GPU meanwhile.

## Competitors

Install TRT-LLM 1.4.0rc2.post1 in its own venv early (large). vLLM 0.32.0 as a second reference (pie ≈ vLLM on the site; confirm). SGLang is far behind on these cells; skip it unless cheap.

## Time plan

- **Deadline:** 16 h from pod start (the pod terminates itself).
- **Early verdict, about 2 h in:** both batching cells on current main, pie vs TRT-LLM (fairness-checked) and vLLM, over the c sweep (at least c64, c128, c256, c512, c1024), ≥3 rounds at the best points, SM clocks. Put it at the top of status.md, with how many lanes each engine actually decodes per step at its best point and TTFT. If both cells are green, check the rest of the curve and the pbatch cells for points red by ≥4.5%; hand over quickly if nothing is red.
- **Attribution** (lanes per step, step time at that batch, prefill interleaving, admission/seating limits; nsys per-component table summing to the delta): by about 5 h.
- **Target hand-over:** about 8 h in. Leave the rest for review rounds.
- Check in with sync.sh at least hourly; notes arrive as `runs/<run>/note-N.md`.

## Expected at hand-over (per the PLAYBOOK)

- `report.md`: baseline on current main (commit, clocks, rounds), TRT-LLM fairness check and its settings, per-component / per-step table, root causes with pie `file:line` and TRT-LLM's `file:line` at 1.4.0rc2.post1, the fix line by line, before/after tables over the c sweep for both models, token equality (full greedy id lists; under concurrency compare base-vs-fix next to base-vs-base), cross-model checks on this card (at least one non-hybrid dense model and one MoE if admission/seating changes, plus Gemma-4-26B c512 tail if admission changes), the other loads (pbatch, raw, TTFT), memory fit, tests (logs in `results/tests/`, GPU suite on the fix build, failures rerun on base), remaining gap broken down, side findings (whether the A100/L40S/A40/H200 cells share the cause), playbook notes.
- `pr.md`: title in pie's style (e.g. "Decode throughput part of the pie vs TensorRT-LLM gap on H100 (…)"), one or two dense paragraphs.
- `results/`: raw JSON and logs behind every number, the engine sha256 per run, analysis scripts.
