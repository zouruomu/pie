# Brief: 20261104-1903-l4-x1-gpt-oss-20b-batching

A short follow-up run. It finishes the pull request for a fix that the previous run, `20261103-2218-l4-x1-gpt-oss-20b-batching`, found and verified. Felix has decided the fix ships, though the site's cells (L4 × gpt-oss-20b batching and pbatch) are already green on main. This run is also the first on new infrastructure, so report the checks at the end of this brief.

## Background

- The previous run's evidence is on `origin/r2g-evidence/20261103-2218-l4-x1-gpt-oss-20b-batching`: `report.md` (read §3 and §5–§12), `pr.md`, `results/`, `scripts/`. Reuse its scripts and harness.
- Its fix is commit `8fc6862cb` on `origin/r2g/20261103-2218-l4-x1-gpt-oss-20b-batching`, on upstream `1b8512478`. It changes one line, `moe.mxfp4_mma_per_from` from 4 to 0, in `crates/engine-cuda/profiles/sm89/nvidia-l4.toml`.
- Measured there, after the rebase: c16 297 → 485 and c8 241 → 280 tok/s (vLLM 0.32.0: 446 and 275); c24 −2.2%; the rest of the curve within spread.
- **What it didn't finish:**
  - the engine-cuda GPU suite (20 of 60 tests done on each build);
  - the final A/B on the final tree. Upstream bumped 20 cargo dependencies after its A/B.
- **Known failure:** `linear::moe_select::the_routed_selects_answer_the_routed_dot` fails on both base and fix, identically, at its pipe-kernel case `(K::Mx, mma, [8, 400, 4, 320, 1024], f, None, 0)`, with every output 3.3895e38. gpt-oss doesn't take that kernel.

## Tasks

1. **The branch.** On `r2g/20261104-1903-l4-x1-gpt-oss-20b-batching` (at upstream main), cherry-pick `8fc6862cb` from `origin/r2g/20261103-2218-l4-x1-gpt-oss-20b-batching`.
2. **One test case.** `the_routed_selects_answer_the_routed_dot` (`crates/kernels-cuda/tests/kernels_cuda/linear/moe_select.rs`) covers the register-dequant select at 1, 6, 8 and 12 routes an expert, but not at 2–3, where the fix moves gpt-oss's c16 decodes.
   - Add one case to the group commented "An Ada's few routes an expert on its 512-code runs", at gpt-oss-20b's c16 shape: 32 experts, 16 tokens, top 4 (64 routes, 2 an expert). For example `(K::MxBias, mma, [32, 16, 4, 320, 2880], t, None, 0)`.
   - Check from the code (`mxfp4_mma_spans`, `mxfp4_run_until`, the test's route pattern) that on sm_89 it takes the run kernel, and adjust it if not.
   - Keep the file's style: one line, plus the smallest change to the group's comment.
   - Fold it into the same single commit. Keep the subject; the body stays empty.
3. **Builds.** The base engine (upstream main) and the fix engine, as in the previous run.
4. **Final A/B** on this tree: 3 alternating rounds of base and fix at c8, c16, c24 and c32, with token ids dumped. Compare tokens as the previous run did (§8), next to base against base. No vLLM runs are needed, since its version hasn't changed.
5. **Tests**, all logged in `results/tests/`:
   - **kernels-cuda `linear::moe_select`** on both builds. The new case must pass on the fix build: the test's panic must still name the pipe-kernel case, which comes after it, and nothing earlier. If the new case fails, stop and report.
   - **engine-cuda GPU suite** (`--test engine_cuda`, `--test-threads=1`), the whole suite on the fix build. Start it as soon as the A/B is done. Then rerun only its failures on the base build.
6. **Write-up.**
   - `report.md`: short. What this run did, the A/B table, the token comparison, the tests with log paths, and anything new. Point to the previous run's report for the rest.
   - `pr.md`: the previous `pr.md`, updated. Use this tree's A/B numbers, the test case, and test lines taken from the logs. Keep its style.

## Time plan

- **Deadline:** 8 h from pod start. **Target hand-over:** about 5 h; the GPU suite takes about 2.5 h on its own.
- **Checkpoint:** the A/B results in status.md, about 1.5 h in.

## Infrastructure checks

Put these in status.md once, early on:
- the **names only** (never values) of environment variables in your Bash tool's shell that match `TOKEN|SECRET|KEY|PASSWORD`;
- whether the Hugging Face download worked;
- whether sync.sh pushed;
- whether you received `note-1.md`. It will arrive within the first hour; acknowledge it.
