# L4 × gpt-oss-20b: finishing the c8/c16 select fix (follow-up run)

Run `20261104-1903-l4-x1-gpt-oss-20b-batching` on one NVIDIA L4 (24 GB, sm_89, 72 W power cap).
It finishes the pull request for the fix that run `20261103-2218-l4-x1-gpt-oss-20b-batching` found
and verified. That run's report (`r2g-evidence/20261103-2218-l4-x1-gpt-oss-20b-batching:report.md`)
holds the attribution (§5), the reasoning behind the line (§6), the whole-curve and cross-model
checks (§3, §9) and the remaining gap against vLLM (§11). They are not repeated here.

* **Fix branch:** `r2g/20261104-1903-l4-x1-gpt-oss-20b-batching`, one commit `f79b846c` on upstream
  `1b851247`.
  * Upstream main has since gained one commit, `48399500` (#1429, `models/glm_5_next.poem` only).
  * Rebase onto it: pending, after the engine-cuda suite, which runs from this tree.
* **Engines:** the A/B's fix engine was built from `2f4f9cf0`. It differs from `f79b846c` only in
  the test case's line and its comment (§2). The engine sources and the profile are identical, so
  the engine is the same.
* **Evidence:** `r2g-evidence/20261104-1903-l4-x1-gpt-oss-20b-batching`.

## 1. What this run did

1. **The commit.** Cherry-picked `8fc6862cb` (the L4 profile's `moe.mxfp4_mma_per_from` 4 → 0) and
   folded one kernel-test case into the same commit. The subject is unchanged and the body is empty
   (§2).
2. **Builds.** Built the base engine (upstream `1b851247`) and the fix engine the same way, in
   separate trees: `uv pip install -e packages/python/server`, a release maturin build.
3. **Final A/B** on this tree: 3 alternating rounds at c8, c16, c24 and c32, token ids dumped (§3,
   §4). vLLM was not rerun, since its version (0.32.0) is unchanged; its numbers below are the
   previous run's.
4. **Tests.** kernels-cuda `linear::moe_select` on both trees, and engine-cuda's whole GPU suite on
   the fix tree (§5).
5. **note-1.** Ran the compute-sanitizer check on the known `moe_select` failure, and traced which
   case that failure is (§6). It is not the pipe kernel.

## 2. The commit

```
 crates/engine-cuda/profiles/sm89/nvidia-l4.toml             | 9 ++++++---
 crates/kernels-cuda/tests/kernels_cuda/linear/moe_select.rs | 3 ++-
```

The profile hunk is `8fc6862cb`'s, unchanged. The test hunk:

```diff
         // An Ada's few routes an expert on its 512-code runs: by route, a
-        // partial row tile.
+        // partial row tile; gpt-oss-20b's c16 GLU, two routes an expert.
         (K::MxBias, mma, [8, 24, 4, 320, 2880], f, None, 0),
+        (K::MxBias, mma, [32, 16, 4, 320, 2880], t, alpha, 0),
```

**Why this case.**

* gpt-oss-20b's c16 decode routes 16 tokens to their top 4 of 32 experts: 64 routes, 2 an expert.
* The fix moves gpt-oss's decodes of 7–24 lanes (28–96 routes, 1–3 an expert) onto the run kernel.
* The test already checked the register-dequant selects at 1, 6, 8 and 12 routes an expert (previous
  report §8), but nothing at 2–3.

**Why this form.**

* The brief's example was `(K::MxBias, mma, [32, 16, 4, 320, 2880], t, None, 0)`. It is the same
  shape, and it also takes the run kernel: `run<4, 4, 0, 4>` in `results/tests/firetrace-fix.log`,
  fire 43.
* But gpt-oss on sm_89 fires the run kernel in two other forms, by the previous run's logs
  (`prev/results/attr/*.log`):
  * `moe_select_mxfp4_mma_run<4, 4, 3, 4>`: the gate/up select, by token, under the clamped SwiGLU
    (`fused.moe_select_glu`);
  * `run<4, 4, 0, 4>`: the down select, by route.
* A no-GLU select by token is neither.
* With `alpha` (the test's `SwigluClampAlpha`), the case is gpt-oss's c16 gate/up select:
  * the instantiation `run<4, 4, 3, 4>`;
  * the expert bias;
  * a K of 2880;
  * 32 experts at 2 routes an expert.
  It is the brief's line with `None` changed to `alpha`.
* With a GLU, `n` = 320 is the output width over a 640-row bank (`moe_select.rs:81`).

**It takes the run kernel on sm_89.** `MxBias` with a GLU goes through `moe::select_glu`
(`moe_select.rs:129`) to `select_mxfp4_mma` (`crates/kernels-cuda/src/linear/moe/select_fp4.rs`).

* **`mxfp4_mma_spans`** (`select_fp4.rs:27-32`) admits the select. The test's `mma` tuning sets
  `mxfp4_mma_from: 1` and `mxfp4_mma_until: 65_536`, and keeps the default
  `mxfp4_mma_per_from = 0`. The select has 64 routes, at ⌈64/32⌉ = 2 an expert.
* **`select_mxfp4_mma`** (`select_fp4.rs:192-207`): `ada_run` holds, since the arch is sm_89,
  K = 2880 = 45 × 64, there are no tensor scales and no cvt. per = 2 ≤ `mxfp4_run_until` (16, the
  default), so the select is not staged and `run` holds.
* **On the GPU:** confirmed by the fire trace of the rerun in §5. Pending.
* **The route pattern** (`moe_select.rs:56`, `(3t + 5s) mod 32`) touches all 32 experts: 4 with 1
  route, 24 with 2 and 4 with 3.

## 3. Final A/B on this tree

**Setup.**

* **Engines** (`results/engines.txt`):
  * base `1b851247`, sha256 `5c7d99c7…`;
  * fix, built from `2f4f9cf0`, sha256 `414293e1…`.
  Each of the 24 JSONs records its engine's sha256, 12 of each.
* **Harness:** the previous run's (`scripts/bench.sh`, `scripts/sweep.sh`; `scripts/ab.sh` here).
  * `tput --num-requests 2N --concurrency N --max-tokens 128`, warmup 2, `--pretokenized-prompts`
    on 225-token chat-rendered prompts, rebuilt with the previous run's `make_prompts.py` (every
    request reports 225 prompt tokens);
  * `--dump-all-token-ids`;
  * one engine at a time on an idle GPU, base then fix within each round, 19:31–20:33Z.
* **CPU:** the bench was pinned to CPUs 0–31 and 64–95. The test binaries compiled on 32–63 and
  96–127 and finished at 19:33:28Z, before the first run's measured window began (19:34:40Z).
* **Clock and power** over each run's measured window (`scripts/clocks_window.py`,
  `results/clocks-window-ab.txt`):
  * Every window draws about 72 W (medians 71.7–72.3 W), so both engines run at the power cap.
  * At c8 and c16 the fix runs at a higher SM clock under the same cap: c8 1,448–1,530 MHz against
    base's 975–1,072, and c16 1,350–1,418 against 1,035–1,065. The by-route select reads each
    expert's codes once a route, the run kernel once. Moving fewer bytes for the same work leaves
    the capped card more clock, so this is part of the mechanism, not a difference between sessions.
  * At c24 and c32 both engines run at 1,290–1,365 MHz.

**Results.** Output tok/s, mean (min–max) of 3 rounds (`results/curve-ab.md`, from `curve.py`):

| shape | base `1b851247` | fix | fix vs base | vLLM 0.32.0 (previous run, 3 rounds) | fix vs vLLM |
|---|---|---|---|---|---|
| c8 | 242.9 (239–245) | 277.4 (273–282) | +14.2% | 275.6 | +0.7% |
| c16 | 301.0 (296–306) | 485.9 (482–488) | +61.4% | 441.9 | +10.0% |
| c24 | 673.6 (668–681) | 664.5 (660–669) | −1.3% | – | – |
| c32 | 835.7 (814–851) | 839.0 (828–853) | +0.4% | 737.0 | +13.8% |

* This agrees with the previous run's rebased pair: c8 241 → 280, c16 297 → 485, c24 −2.2%.
* **c24, the line's cost:** −1.3%. The ranges touch (fix's 669.2 is above base's 668.3), but fix is
  below base in each round's pair: 681.3 vs 660.4, 671.3 vs 669.2, 668.3 vs 664.0. The previous
  run measured −2.2% over 2 rounds.
* **c32:** its 128 routes average 4 an expert, so it was on the run kernel before the fix and keeps
  its kernel. +0.4% is noise, and it serves as the A/B's control.
* **Since the previous A/B:** upstream bumped 20 cargo dependencies (#1315). On this tree the fix's
  c8, c16 and c24 deltas match the previous run's.

## 4. Token equality

Full greedy token-id lists, compared request by request as in the previous report's §8
(`scripts/agree.py`; `results/tokens-ab.txt` holds every pair). Mean prefix share is the share of a
request's 128 tokens before its first difference.

| shape | base vs base (3 pairs) | fix vs fix (3 pairs) | base vs fix, same round (3 pairs) |
|---|---|---|---|
| c8 (16 requests) | 15, 6, 5 identical; prefix 0.97, 0.52, 0.49 | 8, 14, 10; 0.63, 0.90, 0.72 | 6, 6, 3; 0.53, 0.55, 0.41 |
| c16 (32) | 8, 10, 6; 0.49, 0.56, 0.45 | 21, 18, 23; 0.78, 0.72, 0.82 | 11, 3, 8; 0.57, 0.36, 0.55 |
| c24 (48) | 35, 31, 37; 0.82, 0.76, 0.83 | 27, 18, 20; 0.68, 0.60, 0.60 | 24, 25, 12; 0.65, 0.67, 0.52 |
| c32 (64) | 25, 31, 46; 0.61, 0.68, 0.81 | 44, 31, 44; 0.78, 0.66, 0.82 | 34, 43, 32; 0.71, 0.78, 0.67 |

As in the previous run:

* **Reruns vary.** Under concurrency, pie's greedy outputs vary between runs of one engine (prefix
  0.45–0.97 across these reruns).
* **c32, the control** (no kernel change): base against fix, 0.67–0.78, sits inside the reruns'
  range (0.61–0.82).
* **c8 and c16**, where every decode step changes kernel, sit at the low end of the reruns' range or
  a little under it: c8 0.41–0.55 against 0.49–0.97, c16 0.36–0.57 against 0.45–0.82. That is what
  two accumulation orders give.
* **c24** says little: base against fix (0.52–0.67) is about as low as the fix's own reruns
  (0.60–0.68).
* **Numerics:** the kernel test of §5, with this run's new case at 2 routes an expert, checks the
  run kernel against the host's routed dot.

## 5. Tests

On this L4, logs in `results/tests/`.

* **Trees:**
  * fix: `/root/r2g/pie`;
  * base: `/root/src/pie-base` at `1b851247`, without the new case.
* **Builds:** debug builds, compiled with the same commands on both trees (`scripts/tests_build.sh`,
  `tests_moe.sh`, `tests_moe2.sh`, `tests_engine.sh`).
* **Threads:** every test binary ran single-threaded (`--test-threads=1`).
* **The GPU was shared** between test binaries:
  * the engine-cuda suite ran from 20:33Z throughout;
  * beside it, one after another: `moe_select` on fix, then on base (`scripts/after_ab.sh`), then
    the sanitizer, the fire trace and the `moe_select` rerun of §6 and below.
  * The timings below are therefore not comparable with the previous run's.

| suite | fix | base `1b851247` |
|---|---|---|
| kernels-cuda `linear::moe_select` filter: 7 tests, 1 of them timing-only and 1 from `moe_select_mlxu4`. `2f4f9cf0`, with the case's earlier no-GLU form (`kernels-moe-select-{fix,base}.log`) | 5 passed, 1 failed (`the_routed_selects_answer_the_routed_dot`), 1 ignored; 881.5 s | the same 5 passed and the same failure, the same message and values; 879.4 s |
| the same, `f79b846c`, the case's final GLU form, fire trace on (`kernels-moe-select-fix2.log`) | pending | – |
| engine-cuda GPU suite `--test engine_cuda`, 60 tests (`engine-cuda-fix.log`) | pending | failures only: pending |

**The new case passes.**

* On `2f4f9cf0` (no-GLU form), both trees panic with
  `Mx 8x400x4 false 320x1024 None 0 even false: 512000 off: [(0, (3.3895313892515355e38,
  (-928.1639613509178, 216.29599112085998))), …`. That is the previous run's message, with the same
  values.
* `routed` panics at the first case that fails. The fire trace of §6 shows which case that is: the
  first W4A8 case, after the pipe case.
* In that trace the case in its no-GLU form is fires 42–43: `moe_route_order_scan`, then
  `moe_select_mxfp4_mma_run<4, 4, 0, 4>`. Every case before the W4A8 one fires and passes, the new
  case, the pipe case and the three after it included.
* The final GLU form: pending (`kernels-moe-select-fix2.log`).

## 6. note-1: the known `moe_select` failure

**compute-sanitizer finds no launch or CUDA error.** The run was `CUDA_LAUNCH_BLOCKING=1
compute-sanitizer --tool memcheck` on the `2f4f9cf0` build, the test alone, and took 401 s
(`scripts/sanitize.sh`, `results/tests/sanitizer-fix.log`, compute-sanitizer 2026.3.0).

* It ends with "ERROR SUMMARY: 46 errors". All 46 are one benign kind, reported twice for each of
  the 23 kernel loads (once as an API-check failure, once as the error code): `cuFuncGetParamInfo`
  returning `CUDA_ERROR_INVALID_VALUE` past a kernel's last parameter.
* That is pie's own probe for a kernel's parameter count. `param_sizes`
  (`crates/kernels-cuda/src/jit/cache.rs:135-158`) asks for parameters until the driver says there
  are no more.
* There is no memory-access error, no launch failure and no other API error. The test fails exactly
  as without the sanitizer.
* **Coverage:** memcheck covered the cases up to and including the failing one. The cases after it
  never run, with or without the sanitizer.

**The failing case is not the pipe kernel's.** It is the first W4A8 case.

* **The trace.** I reran the test alone with pie's fire trace (`PIE_CUDA_TRACE_FIRES=1`,
  `crates/kernels-cuda/src/jit/ctx.rs:730-733`; `scripts/firetrace.sh`,
  `results/tests/firetrace-fix.log`, 60 fires, 400 s). It shows (fire ordinals; the first fire
  shares a line with the test's name):

  | fires | case (`moe_select.rs`) | outcome |
  |---|---|---|
  | 52–53: `moe_route_order_scan`, `moe_select_mxfp4_mma_pipe<0, 3>` | `(K::Mx, mma, [8, 400, 4, 320, 1024], f, None, 0)`, the pipe case | passes |
  | 54–55: `pipe<3, 3>` | `(K::MxBias, mma, [8, 400, 4, 192, 2048], t, alpha, 2)` | passes |
  | 56–57: `staged<8, 3, 0, false, 8>` | `(K::Mx, mma, [8, 48, 4, 384, 1024], t, None, 0)` | passes |
  | 58–60: `staged<8, 3, 3, …>`, then `staged<8, 3, 0, …>` with no route scan (flag 8's "select after", in the kept order) | `(K::MxBias, mma, [8, 400, 4, 192, 2880], t, alpha, 8)` | passes |
  | none | `(K::Mx, w4a8, [8, 400, 4, 320, 1024], f, None, 0)` (`moe_select.rs:291`; 290 on upstream) | **fails**: every output is the fill |

* **Why the W4A8 case fails on sm_89.**
  * `select_mxfp4_w4a8` returns `Ok(false)` off sm_120a: `w4a8_takes`
    (`select_fp4.rs:413-422`) requires `features().warp_block_scale`.
  * The test accepts that: `assert_eq!(took, w4a8_here(), "the arm takes the fire")`,
    `moe_select.rs:139`.
  * It then still compares `y` against the routed dot (`moe_select.rs:185`). `y` was never written,
    so all 1,600 × 320 = 512,000 outputs read back as the 0x7f7f fill (`moe_select.rs:113`):
    3.3895e38.
* **Why it looked like the pipe case.** The description `what` (`moe_select.rs:53`) leaves out the
  tuning, so this case prints exactly as the pipe case does: `Mx 8x400x4 false 320x1024 None 0`.
* **Where it comes from.** The W4A8 cases came with #1274 (`fc7ade063`, "sm_120a: MXFP4 selects of
  W4A8 deployments on the block-scaled mma.sync"). The group's comment says "(elsewhere the bf16
  arms)", but on a card without the block-scaled mma nothing in the W4A8 branch fires a bf16 arm.
* **For Anton.**
  * This is a test failure, not a kernel failure, on every card without `warp_block_scale`, which
    only sm_120a and sm_121a have (`crates/kernels-cuda/src/jit/root.rs:252`). The cases after it
    (the TMA and warp-specialized W4A8 cases, and the rest) do not run there.
  * The pipe kernel passes on sm_89.
  * Not fixed here, as note-1 asks.
* **This corrects the previous run's report.** Its §10 and §12.11, and the brief, put the failure on
  the pipe kernel. The panic names a description, and two cases share it.

## 7. New in this run

* **Infrastructure** (status.md has the details):
  * The only environment variable in the agent's Bash shell matching `TOKEN|SECRET|KEY|PASSWORD` is
    `CLAUDE_CODE_MESSAGING_TOKEN` (name only). It was removed from every build, bench and test
    process: `env -u` on the build commands, `unset` in the scripts.
  * The Hugging Face download worked.
  * sync.sh pushed.
  * note-1 arrived at the first sync.
* **Pod setup, for future runs:**
  * The bench venv needs `transformers` for `--pretokenized-prompts`. The first A/B attempt at 19:31Z
    failed at its first run with `ModuleNotFoundError` and was restarted from scratch. Its files were
    deleted, so nothing records it.
  * A maturin build that starts while the nightly toolchain is still installing fails in
    `crates/builtins/build.rs` ("the 'cargo' binary … is not applicable"). Install
    `nightly-2026-10-18` with the `wasm32-wasip3` target before any build.
  * Building the bench inferlet rewrites `examples/Cargo.lock` (wasm-encoder 0.259 → 0.262). The fix
    tree carried that uncommitted change while its engine was built. The file is outside the
    engine's sources; it was restored before the amend.
* **The known `moe_select` failure is a test issue, not the pipe kernel** (§6). It needs a fix by
  the test's owner. On any card without sm_120a's block-scaled mma, it hides every case after the
  first W4A8 one.
* **Playbook note:** a kernel test's panic names a description, and two cases can share it. Before
  attributing a failure to a kernel, run the test with `PIE_CUDA_TRACE_FIRES=1` and match the fires
  to the case list.
* **Independent review:** a fresh-context reviewer recomputed every number here. Its findings were
  folded in:
  * c24's ranges touch;
  * the clocks over the measured windows;
  * the fire ordinals;
  * the sanitizer's 23 + 23;
  * the new case's GLU form;
  * the GPU sharing between test binaries.

## 8. Files

* `scripts/`:
  * `ab.sh`, with `sweep.sh`, `bench.sh` and `make_prompts.py` from the previous run;
  * `tests_build.sh`, `tests_moe.sh`, `tests_moe2.sh`, `tests_engine.sh`, `after_ab.sh`;
  * `sanitize.sh`, `firetrace.sh`;
  * analysis: `curve.py`, `summ.py`, `clocks.py`, `clocks_window.py`, `agree.py`, `tokeq.py`.
* `results/`:
  * `ab/`: the 24 runs, with `.json`, `.log` and `.clk` for each;
  * `curve-ab.md`, `summ-ab.txt`, `clocks-ab.txt` (whole runs, mostly load), `clocks-window-ab.txt`
    (measured windows), `tokens-ab.txt`, `engines.txt`;
  * `tests/`.
