# Status: 20261104-1903-l4-x1-gpt-oss-20b-batching

**Updated:** 2026-11-04 19:30Z (pod start 19:16Z; deadline about 03:16Z; target hand-over about 00:15Z)

**Phase:** builds (base engine, fix engine, bench inferlet).

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

**Running:** cargo/maturin builds of base (`/root/src/pie-base`, upstream main) and fix
(`/root/r2g/pie`); the wasm32-wasip3 bench inferlet.

**Blockers:** none.

**ETA hand-over:** about 00:00Z.

## Infrastructure checks

* **Environment variables in my Bash tool's shell matching `TOKEN|SECRET|KEY|PASSWORD` (names
  only):** `CLAUDE_CODE_MESSAGING_TOKEN`. That is the only match. I strip it (`env -u`) from every
  build, bench and test process I start.
* **Hugging Face download:** worked. `hf download openai/gpt-oss-20b` (excluding `original/` and
  `metal/`) gave all 3 safetensors shards, 13 GB, within a few minutes of the start. The blobs sit
  under `~/.cache/huggingface/hub/blobs/<xx>/`, a content-addressed layout.
* **sync.sh pushed:** pending (first sync at 19:30Z).
* **note-1.md:** not yet received.

## Orchestrator notes

* none yet
