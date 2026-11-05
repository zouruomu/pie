# Status: 20261104-1903-l4-x1-gpt-oss-20b-batching

**READY FOR REVIEW** — 2026-11-05 01:39Z. Pod created about 19:03Z; deadline taken as 03:03Z.

**Fix branch:** `r2g/20261104-1903-l4-x1-gpt-oss-20b-batching`, one commit `8e8b19b0` on upstream
`48399500`, the newest. The subject is unchanged; the body is empty. It changes the L4 profile line
(`moe.mxfp4_mma_per_from` 4 → 0) and adds one kernel-test case,
`(K::MxBias, mma, [32, 16, 4, 320, 2880], t, alpha, 0)`: gpt-oss-20b's c16 gate/up select, on
`moe_select_mxfp4_mma_run<4, 4, 3, 4>`, the instantiation gpt-oss itself fires. It passes.

**Results** (report.md has the details and log paths; pr.md is updated):

* **A/B**, 3 rounds on `1b851247`: c8 242.9 → 277.4 (+14.2%), c16 301.0 → 485.9 (+61.4%), c24
  −1.3%, c32 +0.4%.
* **After the rebase**, 3 rounds: c8 +16.1%, c16 +65.0%, c24 −1.1%, c32 −1.3% (base's own c32 runs
  span 798–852).
* **Tokens:** base against fix sits at the low end of the reruns' range at c8/c16, and inside it at
  c32 (the control). After the rebase it sits inside the reruns' range at every shape.
* **kernels-cuda `moe_select`:** the same 5 pass on both builds. The sixth fails identically on
  both, at a W4A8 case, not the pipe kernel's (below).
* **engine-cuda GPU suite** (whole suite, single-threaded, dev build at opt-level 2 with debug
  assertions): 55 passed, 4 failed, 1 ignored.
  * The 4 failures ran short of device memory beside my other test processes.
  * Each passes alone on the GPU, on fix and on base.
  * The debug build was too slow: over an hour in single tests.
* **note-1:** compute-sanitizer finds no launch or CUDA error, only pie's own `cuFuncGetParamInfo`
  probe.
  * A fire trace shows that the pipe kernel passes on sm_89.
  * The failure is the first W4A8 case (#1274): its select declines off sm_120a, and the test
    compares an output it never wrote. Its description matches the pipe case's.
  * This is a test bug, for Anton, and it corrects the previous run's attribution.
* **Self-review:** a fresh-context agent recomputed the numbers. Its findings are folded in,
  including the change of the case to gpt-oss's GLU form.

**Running:** nothing.

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
