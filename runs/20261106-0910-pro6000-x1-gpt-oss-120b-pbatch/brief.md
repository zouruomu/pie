# Brief: 20261106-0910-pro6000-x1-gpt-oss-120b-pbatch

## The cells (RTX PRO 6000 Blackwell Server Edition, sm_120, 96 GB; site hardware `pro6000-x1`)

Two prefill-batch (`pbatch`) cells, scheme `mxfp4`, full models (not miniatures). The best competitor is TensorRT-LLM 1.4.0rc2.post1, whose rows were all measured on 2026-11-05 (site group `a40-trtllm`), two to three days after pie's.

**gpt-oss-120b × pbatch: pie −67.5% vs TRT-LLM** (gain −0.6755; pie commit `ee9098f60`, measured 2026-11-03 14:05; stale tag `pro6000-fill`)

| engine | version | prefill tok/s (best pf point) | TTFT p50 ms | measured |
|---|---|---|---|---|
| pie | `ee9098f60` | 13776.67 | 2766.88 | 2026-11-03 14:05 |
| vLLM | 0.32.0 | 22268.49 | 3529.84 | 2026-11-03 14:12 |
| SGLang | 0.5.22 | 21194.41 | 3781.37 | 2026-11-03 14:20 |
| TRT-LLM | 1.4.0rc2.post1 | **42448.66** | 887.54 | 2026-11-05 20:02 |

**gpt-oss-20b × pbatch: pie −30.2% vs TRT-LLM** (gain −0.3018; pie commit `ea800efeb`, measured 2026-11-02 16:40; stale tag `pro6000-full`)

| engine | version | prefill tok/s | TTFT p50 ms | measured |
|---|---|---|---|---|
| pie | `ea800efeb` | 49789.64 | 188.23 | 2026-11-02 16:40 |
| vLLM | 0.32.0 | 39560.97 | 540.67 | 2026-10-29 18:20 |
| SGLang | 0.5.22 | 31656.11 | 689.08 | 2026-10-29 18:26 |
| TRT-LLM | 1.4.0rc2.post1 | **71315.24** | 513.31 | 2026-11-05 19:55 |

What the numbers suggest (hypotheses to test, not conclusions):
- **120b has a pie-specific problem on this card beyond TRT-LLM:** pie trails vLLM and SGLang too (−38%), and pie's 120b/20b ratio is 0.28 where TRT-LLM's is 0.60 (36 vs 24 layers, 128 vs 32 experts predicts ~0.6). Candidates: the MXFP4 bf16 copies not fitting beside 61 GiB of weights (#1401, `crates/engine-cuda/src/weights.rs`), so wide prefill selects fall back to a slower path; fire size / working set; the 128-expert select shape.
- **Both models trail TRT-LLM at large M only:** at concurrency 1 (raw prefill) pie leads TRT-LLM on this card (20b 24159 vs 13991; 120b 9971 vs 10755 on an older pie). Large-M MoE GEMMs are compute-bound, so check what precision TRT-LLM's MoE runs on sm_120 (W4A8, MXFP4 weights × MXFP8 activations on the block-scaled MMA, would explain ~2× over pie's register-dequant bf16 `mma.sync` from `moe.mxfp4_mma_from = 28`, `crates/engine-cuda/profiles/sm120.toml`). If so, decide fairness explicitly: pie has W4A8 MXFP4 paths (Hopper: #1374, #1387; sm_120a: `select_fp4_sm120.cu`). A pie W4A8 path is a legitimate fix only if its numerics are acceptable and documented; say what precision every engine ran.
- **Check TRT-LLM's numbers are fair before chasing them:** cold prompts (TRT-LLM's KV block reuse is on by default, and 256 prompts sharing a chat-template prefix would skip part of the prefill), the same 512-token prompts, the same 1 output token, its best pf point. Find the TRT-LLM harness the site used (anton-mel/stats on GitHub; pie's `scripts/bench` has no TRT-LLM script). On H200 TRT-LLM equals pie on 20b pbatch (60737 vs 60733), so the sm_120 lead is not a generic harness artifact, but verify.

## Workloads (site `workloads` table)

- **prefill_batch** (these cells): 512-token prompts, 1 output token, sweep pf4, pf8, pf16, pf32, pf64, pf128, pf256, pf512, pf1024 (`pf256-512` is the default point). prefill tok/s is scored; the site takes each engine's best point. Mid-curve points where pie trails the best competitor at the same point by ≥4.5% are in scope.
- throughput (batching): 128 in, 128 out, sweep c4 … c1024. raw: 1 request, 512 in, 256 out.
- Exact flags: `scripts/bench` and `.github/workflows/evals.yml`.

## Related red cells (same pattern; check alongside where cheap)

On this card (measure these on the pod):
- PRO 6000 × gpt-oss-120b × batching: pie 6214.56 vs TRT-LLM 8877.38 (−30.0%), TTFT p50 20864 vs 532 ms; vLLM 8536.53. Likely shares the 120b cause.
- PRO 6000 × gpt-oss-20b × batching is green on the site (19177 vs TRT-LLM 19578, −2.0%); raw prefill green for 20b. Don't give them back.

Other sm_120 cards (not on this pod; say in the report whether your cause applies to them):
- RTX 5090 gpt-oss-20b pbatch −44.2% (32385 vs 57987), gpt-oss-120b (24-layer mini) pbatch −43.6%, batching −36.4% / −30.6%, prefill −14.6% / −9.7%.
- RTX PRO 4500 gpt-oss-20b pbatch −37.5%, 120b (mini) pbatch −28.5%, batching −20.0% / −19.4%, 120b prefill −9.1%.

## Commits on main since the measured commits (checked 2026-11-06 09:10Z, main at dc6282d76)

Nothing names gpt-oss prefill on sm_120. Read these first, since they could already change the cells:
- **`c1f56bb` sm_120a MXFP4 select: compile with device as the default execution space (#1430, 11-04).** `crates/kernels-cuda/src/jit/root.rs` adds `linear/moe/select_fp4_sm120.cu` to the units compiled with `--device-as-default-execution-space`. If that unit failed to JIT before, the sm_120a select declined and the site's pie numbers ran a fallback. Find out in your first hour whether this changes the cells (compare the profile's chosen path at ee9098f60/ea800efeb vs main if cheap; at least check which select runs on main).
- `882f246` MXFP4 bf16 copies only where they fit the free memory (#1401, 11-03, after ee9098f60): directly relevant to 120b on 96 GB.
- `8ca5177` Prefill throughput part of the pie vs vLLM gap on RTX 4090 (bf16 split sums, finer prefill lattice) (#1421): shows how the maintainers attack prefill throughput.
- `b396a8e` Hopper MXFP4 experts under MXFP8 activations take the FP8 copy on wide selects (#1387), `55c71fc` H100 W4A8 MXFP4 selects on a block-FP8 copy past 256 routes (#1374): the Hopper W4A8 precedent.
- `e4bb499` runtime: a piece writes only the working sets with rows in it (#1436); `a654750` vllm_bench `--max-num-seqs` cap (#1437).
- sm120 profiles last changed in #1357, #1358 (RTX 5090), #1359/#1361/#1362 (PRO 4500), #1344 (sm_120a NVFP4 W4A4 selects): none for gpt-oss.
- Open PRs on pie-project/pie: none on gpt-oss, MXFP4 or sm_120 (#1484 is TPU GDN).

## Ledger and other runs

- No earlier run on these cells. From the L4 gpt-oss runs: `the_routed_selects_answer_the_routed_dot` fails off sm_120a/121a at #1274's first W4A8 case; on sm_120 it should pass, so a failure there is real. Build the engine-cuda GPU suite at `CARGO_PROFILE_DEV_OPT_LEVEL=2` and run nothing else on the GPU while it runs.
- The L4 run found `moe.mxfp4_mma_per_from` (by-route MXFP4 select thresholds) mattered a lot for 1–4 routes an expert; at pbatch every expert has hundreds of routes, so the wide-select path is what matters here.
- Concurrently active: A100 SXM × gemma-4-31b batching. Its fix lifts #1368's admission cap (`crates/runtime/src/bootstrap.rs`) where pools seat to the end, which includes the PRO 6000's profile. That bears on the 120b **batching** cell (not pbatch). Don't fix the admission cap here; if the 120b batching gap is the cap, say so and leave it to that run.
- Competitor environments: install TRT-LLM 1.4.0rc2.post1 in its own venv (it is large; start early). vLLM 0.32.0 for a second reference on 120b.

## Time plan

- **Deadline:** 20 h from pod start (the pod terminates itself).
- **Early verdict:** about 2–2.5 h in (two checkpoints and the TRT-LLM install take time): both pbatch cells on current main, pie vs TRT-LLM (and vLLM for 120b) over the pf sweep, ≥3 rounds at the best points, SM clocks, plus what precision TRT-LLM's MoE runs. Put it at the top of status.md. If both are green, check the batching curve and pf points for mid-curve red; hand over quickly if nothing is red.
- **Attribution** (nsys, per-component table for one large pf point, pie vs TRT-LLM, summing to the delta): by about 5 h.
- **Target hand-over:** about 10 h in.
- Check in with sync.sh at least hourly; notes arrive as `runs/<run>/note-N.md`.

## Expected at hand-over (per the PLAYBOOK)

- `report.md`: baseline on current main (commit, clocks, rounds), the fairness check of TRT-LLM's numbers and the precision each engine ran, the per-component table, root causes with pie `file:line` and TRT-LLM's `file:line` at 1.4.0rc2.post1, the fix line by line, before/after tables over the pf sweep for both models, token equality (pbatch has 1 output token: also compare greedy ids on a raw and a batching run for any change that alters numerics, and quantify numerics if a kernel or precision changes), cross-model checks on this card (at least one non-gpt-oss MoE and one dense model if shared paths change), the other loads (batching curve, raw, TTFT), memory fit, tests (logs in `results/tests/`, GPU suite on the fix build, failures rerun on base), remaining gap broken down, side findings (including whether RTX 5090 / PRO 4500 share the cause), playbook notes.
- `pr.md`: title in pie's style (e.g. "Prefill throughput part of the pie vs TensorRT-LLM gap on RTX PRO 6000 (…)"), one or two dense paragraphs.
- `results/`: raw JSON and logs behind every number, the engine sha256 per run, analysis scripts.
- If the cause needs a new sm_120 kernel that won't fit this run, say so early in status.md with the measured size of the win, and hand over the attribution instead of a half-built kernel.
