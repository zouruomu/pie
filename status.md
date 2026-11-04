# Status: 20261104-1903-l4-x1-gpt-oss-20b-batching

**Updated:** 2026-11-04 20:04Z (pod created about 19:03Z, bootstrap 19:16Z; deadline taken as 03:03Z; target hand-over about 00:05Z)

**Phase:** final A/B (since 19:31Z).

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

**Running:**

* `scripts/ab.sh`: 3 alternating rounds of base and fix at c8, c16, c24 and c32, token ids dumped,
  into `results/ab/`. ETA about 20:35Z.
* Test binaries for kernels-cuda and engine-cuda: compiled on both trees (`results/tests/build-*.log`,
  EXIT=0).

**Partial A/B (round 1 complete, round 2 under way; output tok/s, `curve.py`):**

| shape | base (n) | fix (n) | fix vs base |
|---|---|---|---|
| c8 | 244.8 (2) | 277.3 (1) | +13.2% |
| c16 | 298.3 (2) | 482.2 (1) | +61.7% |
| c24 | 676.3 (2) | 660.4 (1) | −2.3% |
| c32 | 813.6 (1) | 852.7 (1) | +4.8% (single runs; c32's selects don't change kernel, so this is not yet read as an effect) |

**Blockers:** none.

**ETA hand-over:** about 00:00Z.

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
