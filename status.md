# Status: 20261104-1903-l4-x1-gpt-oss-20b-batching

**Updated:** 2026-11-04 21:52Z (pod created about 19:03Z, bootstrap 19:16Z; deadline taken as 03:03Z; target hand-over about 00:05Z)

**Phase:** tests (since 20:33Z). The A/B is done (19:31–20:33Z).

* **Engines** (`results/engines.txt`):
  * base `1b851247`, sha256 `5c7d99c7…`;
  * fix `2f4f9cf0`, sha256 `414293e1…`.

**Last result:**

* **The commit is now `f79b846c`** (pushed; on upstream `1b851247`).
* **The self-review** (a fresh-context agent) recomputed the numbers. It found errors in my drafts,
  now fixed in report.md and pr.md:
  * c24's ranges touch, so I cite the per-round pairs;
  * clocks over the measured windows, which show both engines at the 72 W cap, and the fix's higher
    clock at c8/c16;
  * the fire ordinals;
  * the sanitizer's 23 + 23;
  * the GPU sharing between test binaries.
* **It also found that the brief's example case (no GLU, by token) is a form gpt-oss never fires.**
  gpt-oss's gate/up select on sm_89 is `run<4, 4, 3, 4>` (GLU, by token), and its down select is
  `run<4, 4, 0, 4>` by route. So the case is now
  `(K::MxBias, mma, [32, 16, 4, 320, 2880], t, alpha, 0)`: the brief's line with `alpha`, gpt-oss's
  c16 gate/up select. The comment reads "partial row tile; gpt-oss-20b's c16 GLU, two routes an
  expert."
* **The engine is unchanged by this:** only the test file differs from `2f4f9cf0`, the A/B's fix
  build.
* **The previous form passed:** fires 42–43 in `results/tests/firetrace-fix.log`, `run<4, 4, 0, 4>`.
  The new form passes too (`results/tests/kernels-moe-select-fix2.log`, fire trace on).
  It is fires 42–43 of the routed test, `fused.moe_select_glu` → `moe_select_mxfp4_mma_run<4, 4, 3, 4>`,
  and the panic is again the W4A8 case's, with the same values.

**A/B checkpoint (done; `results/ab/`, `results/curve-ab.md`):** 3 alternating rounds, output tok/s, mean (min–max).

| shape | base `1b851247` | fix `2f4f9cf0` | fix vs base | vLLM 0.32.0 (previous run) |
|---|---|---|---|---|
| c8 | 242.9 (239–245) | 277.4 (273–282) | +14.2% | 275.6 |
| c16 | 301.0 (296–306) | 485.9 (482–488) | +61.4% | 441.9 |
| c24 | 673.6 (668–681) | 664.5 (660–669) | −1.3% | – |
| c32 | 835.7 (814–851) | 839.0 (828–853) | +0.4% | 737.0 |

* Every JSON records its engine's sha256: 12 base and 12 fix.
* At-cap SM clock medians are 1,200–1,230 MHz in every run (`results/clocks-ab.txt`).
* **Tokens** (`results/tokens-ab.txt`, mean prefix share per pair):

  | shape | base vs base | fix vs fix | base vs fix, same round |
  |---|---|---|---|
  | c8 | 0.49–0.97 | 0.63–0.90 | 0.41–0.55 |
  | c16 | 0.45–0.56 | 0.72–0.82 | 0.36–0.57 |
  | c24 | 0.76–0.83 | 0.60–0.68 | 0.52–0.67 |
  | c32 | 0.61–0.81 | 0.66–0.82 | 0.67–0.78 |

  c32 keeps its kernel and serves as the control: base against fix falls inside its reruns' range.

**kernels-cuda `linear::moe_select`, fix tree: the new case passes.**

* 5 passed, 1 failed, 1 ignored, in 881.5 s (`results/tests/kernels-moe-select-fix.log`).
* The one failure, `the_routed_selects_answer_the_routed_dot`, panics only at the pipe-kernel case
  `Mx 8x400x4 false 320x1024 None 0`, which comes after the new case. Its values are the previous
  run's: 3.3895e38 against −928.16….
* The cases run in order and the first failing one panics, so every earlier case passed, the new
  `MxBias 32x16x4 true 320x2880` included.

**kernels-cuda `linear::moe_select`, base tree:** the same 5 passed, and the same failure at the same
pipe case with the same values, in 879.4 s (`results/tests/kernels-moe-select-base.log`).

**note-1 (done; report §6):**

* compute-sanitizer memcheck with `CUDA_LAUNCH_BLOCKING=1` took 401 s. It found no launch or CUDA
  error, only pie's own `cuFuncGetParamInfo` parameter-count probe (`jit/cache.rs:136-158`).
* A fire trace (`PIE_CUDA_TRACE_FIRES=1`, `results/tests/firetrace-fix.log`) shows the failing case
  is not the pipe kernel's.
  * The pipe case fires `moe_select_mxfp4_mma_pipe<0, 3>` and passes, as do the 4 cases after it.
  * The failure is the first W4A8 case `(K::Mx, w4a8, [8, 400, 4, 320, 1024], f, None, 0)` (#1274),
    which prints the same description. Off sm_120a its select declines (`Ok(false)`, which the test
    accepts), and the test then compares the never-written output: the 0x7f7f fill.
  * This is a test bug, for Anton. It corrects the previous run's attribution.
* The trace also shows the new case firing `moe_select_mxfp4_mma_run<4, 4, 0, 4>`.

**Running:**

* the engine-cuda GPU suite on the fix tree (`results/tests/engine-cuda-fix.log`). Then its
  failures, if any, rerun on base;
* engine-cuda tests 22–60 in a second single-threaded process, since 21:53Z
  (`results/tests/engine-cuda-fix-b.log`). The first process has been in test 21
  (`a_joined_projection_answers_what_its_legs_answer`, catalog-wide tracing in a debug build) since
  about 20:58Z.

At hand-over: rebase onto `48399500` (#1429, `models/glm_5_next.poem` only) once the suite is done,
since the suite runs from this tree.

**Blockers:** none.

**ETA hand-over:** about 23:45Z. The engine-cuda suite sets the pace.

## Infrastructure checks

* **Environment variables in my Bash tool's shell matching `TOKEN|SECRET|KEY|PASSWORD` (names
  only):** `CLAUDE_CODE_MESSAGING_TOKEN`. That is the only match. I strip it (`env -u`) from every
  build, bench and test process I start.
* **Hugging Face download:** worked. `hf download openai/gpt-oss-20b` (excluding `original/` and
  `metal/`) gave all 3 safetensors shards, 13 GB, within a few minutes of the start. The blobs sit
  under `~/.cache/huggingface/hub/blobs/<xx>/`, a content-addressed layout.
* **sync.sh pushed:** yes. The first sync at 19:25Z pushed `r2g-evidence/20261104-1903-l4-x1-gpt-oss-20b-batching` (remote head `4507bb7`).
* **note-1.md:** received at the first sync (19:25Z) and acknowledged below.

## Orchestrator notes

* **note-1.md (19:25Z), acknowledged.** Item 2: after both `moe_select` runs, if it takes under 20 minutes, I will rerun `the_routed_selects_answer_the_routed_dot` on the fix build with `CUDA_LAUNCH_BLOCKING=1` under `compute-sanitizer`, and record in report.md whether a launch or CUDA error appears and what it says. No fix (it goes to Anton).
