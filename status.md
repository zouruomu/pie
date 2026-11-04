# Status: 20261104-1903-l4-x1-gpt-oss-20b-batching

**Updated:** 2026-11-04 20:45Z (pod created about 19:03Z, bootstrap 19:16Z; deadline taken as 03:03Z; target hand-over about 00:05Z)

**Phase:** tests (since 20:33Z). The A/B is done (19:31–20:33Z).

* **Engines** (`results/engines.txt`):
  * base `1b851247`, sha256 `5c7d99c7…`;
  * fix `2f4f9cf0`, sha256 `414293e1…`.

**Last result:** the branch `r2g/20261104-1903-l4-x1-gpt-oss-20b-batching` is one commit, `2f4f9cf0`,
on upstream `1b851247`. It is `8fc6862cb` cherry-picked, with the new kernel-test case folded in. Same
subject, empty body.

* The new case: `(K::MxBias, mma, [32, 16, 4, 320, 2880], t, None, 0)`, in the
  "An Ada's few routes an expert on its 512-code runs" group.
* From the code, on sm_89 it takes the run kernel:
  * `mxfp4_mma_spans`: `mma` sets from 1 and until 65,536, and per_from 0 is the default. So 64
    routes are admitted.
  * `select_mxfp4_mma`: `ada_run` holds (sm_89, K 2880 = 45 × 64, no tensor scales, no cvt), and
    per = ⌈64/32⌉ = 2 ≤ `mxfp4_run_until` (16), so the select is not staged and `run` holds.
  * The test's route pattern `(3t + 5s) mod 32` touches all 32 experts: 4 with 1 route, 24 with 2
    and 4 with 3.

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

**Running** (since 20:33Z, side by side):

* engine-cuda GPU suite on the fix tree, `results/tests/engine-cuda-fix.log`, ETA about 23:00Z;
* kernels-cuda `linear::moe_select`, fix then base, `results/tests/kernels-moe-select-{fix,base}.log`,
  about 15 min each;
* then compute-sanitizer (note-1).

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
