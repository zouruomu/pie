# L4 × gpt-oss-20b: finishing the c8/c16 select fix (follow-up run)

Run `20261104-1903-l4-x1-gpt-oss-20b-batching` on one NVIDIA L4 (24 GB, sm_89, 72 W power cap).
It finishes the pull request for the fix that run `20261103-2218-l4-x1-gpt-oss-20b-batching` found
and verified. That run's report (`r2g-evidence/20261103-2218-l4-x1-gpt-oss-20b-batching:report.md`)
holds the attribution (§5), the reasoning behind the line (§6), the whole-curve and cross-model
checks (§3, §9) and the remaining gap against vLLM (§11). They are not repeated here.

* **Fix branch:** `r2g/20261104-1903-l4-x1-gpt-oss-20b-batching`, one commit `2f4f9cf0` on upstream
  `1b851247`, which is still upstream main (sync.sh reports 0 new commits).
* **Evidence:** `r2g-evidence/20261104-1903-l4-x1-gpt-oss-20b-batching`.

## 1. What this run did

1. Cherry-picked `8fc6862cb` (the L4 profile's `moe.mxfp4_mma_per_from` 4 → 0) and folded one
   kernel-test case into the same commit, which keeps its subject and an empty body (§2).
2. Built the base engine (upstream `1b851247`) and the fix engine (`2f4f9cf0`) the same way:
   `uv pip install -e packages/python/server`, a release maturin build, in separate trees.
3. Ran the final A/B on this tree: 3 alternating rounds at c8, c16, c24 and c32, token ids dumped
   (§3, §4). vLLM was not rerun, since its version (0.32.0) is unchanged; its numbers below are the
   previous run's.
4. Ran kernels-cuda `linear::moe_select` on both trees, and engine-cuda's whole GPU suite on the fix
   tree (§5).
5. Ran note-1's compute-sanitizer check on the known `moe_select` failure (§6).

## 2. The commit

```
 crates/engine-cuda/profiles/sm89/nvidia-l4.toml             | 9 ++++++---
 crates/kernels-cuda/tests/kernels_cuda/linear/moe_select.rs | 3 ++-
```

The profile hunk is `8fc6862cb`'s, unchanged. The test hunk:

```diff
         // An Ada's few routes an expert on its 512-code runs: by route, a
-        // partial row tile.
+        // partial row tile; by token at two (gpt-oss-20b's c16 decode).
         (K::MxBias, mma, [8, 24, 4, 320, 2880], f, None, 0),
+        (K::MxBias, mma, [32, 16, 4, 320, 2880], t, None, 0),
```

**Why this case.** gpt-oss-20b's c16 decode routes 16 tokens to their top 4 of 32 experts, so 64
routes average 2 an expert. Before this case, the test checked the register-dequant select at 1, 6,
8 and 12 routes an expert (previous report §8), but nothing at 2–3, where the fix moves c8–c24.

**It takes the run kernel on sm_89.** `MxBias` goes through `matmul_select_bias`, then
`matmul_select_mxfp4` (`crates/kernels-cuda/src/linear/moe/select_fp4.rs:855`), then
`mxfp4_mma_worth`. From there:

* **`mxfp4_mma_spans`** (`select_fp4.rs:27-32`) admits the select. The test's `mma` tuning sets
  `mxfp4_mma_from: 1` and `mxfp4_mma_until: 65_536`, and keeps the default
  `mxfp4_mma_per_from = 0`. The select has 64 routes, at ⌈64/32⌉ = 2 an expert.
* **`select_mxfp4_mma`** (`select_fp4.rs:192-207`): `ada_run` holds, since the arch is sm_89,
  K = 2880 = 45 × 64, there are no tensor scales and no cvt. per = 2 ≤ `mxfp4_run_until` (16, the
  default), so the select is not staged and `run` holds. That is `moe_select_mxfp4_mma_run`, the
  kernel the fix gives gpt-oss's c8–c24 decodes.
* **The route pattern** (`moe_select.rs:56`, `(3t + 5s) mod 32`) touches all 32 experts: 4 with 1
  route, 24 with 2 and 4 with 3.
* **Rows:** by token, as gpt-oss's gate/up select reads them. The existing case covers by route.
  n = 320 leaves a partial 128-row tile.

## 3. Final A/B on this tree

**Setup.**

* **Engines** (`results/engines.txt`):
  * base `1b851247`, sha256 `5c7d99c7…`;
  * fix `2f4f9cf0`, sha256 `414293e1…`.
  Each of the 24 JSONs records its engine's sha256, 12 of each.
* **Harness:** the previous run's (`scripts/bench.sh`, `scripts/sweep.sh`; `scripts/ab.sh` here).
  * `tput --num-requests 2N --concurrency N --max-tokens 128`, warmup 2, `--pretokenized-prompts`
    on 225-token chat-rendered prompts, rebuilt with the previous run's `make_prompts.py` (every
    request reports 225 prompt tokens);
  * `--dump-all-token-ids`;
  * one engine at a time on an idle GPU, base then fix within each round, 19:31–20:33Z.
  * The bench was pinned to CPUs 0–31 and 64–95. The test binaries compiled on 32–63 and 96–127,
    finishing at 19:33Z, during the first run (base c8, round 1: 244.2, inside base c8's 239–245).
* **Clock:** the at-cap SM clock median (samples above 65 W) was 1,200–1,230 MHz in every run, base and fix
  alike (`results/clocks-ab.txt`).

**Results.** Output tok/s, mean (min–max) of 3 rounds (`results/curve-ab.md`, from `curve.py`):

| shape | base `1b851247` | fix `2f4f9cf0` | fix vs base | vLLM 0.32.0 (previous run, 3 rounds) | fix vs vLLM |
|---|---|---|---|---|---|
| c8 | 242.9 (239–245) | 277.4 (273–282) | +14.2% | 275.6 | +0.7% |
| c16 | 301.0 (296–306) | 485.9 (482–488) | +61.4% | 441.9 | +10.0% |
| c24 | 673.6 (668–681) | 664.5 (660–669) | −1.3% | – | – |
| c32 | 835.7 (814–851) | 839.0 (828–853) | +0.4% | 737.0 | +13.8% |

* This agrees with the previous run's rebased pair: c8 241 → 280, c16 297 → 485, c24 −2.2%.
* **c24, the line's cost:** −1.3% here, with ranges 668–681 and 660–669 apart. It is the same sign
  as the previous run's −2.2% (2 rounds), and smaller.
* **c32:** its 128 routes average 4 an expert, so it was on the run kernel before the fix and keeps
  its kernel. +0.4% is noise, and it serves as the A/B's control.
* **Since the previous A/B:** upstream bumped 20 cargo dependencies. On this tree the fix's c8,
  c16 and c24 deltas match the previous run's, so the bumps did not change the result.

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

* **Reruns vary.** Under concurrency, pie's greedy outputs vary between runs of one engine (base
  against base: prefix 0.45–0.97).
* **c32, the control** (no kernel change): base against fix, 0.67–0.78, sits inside the reruns'
  range (0.61–0.82).
* **c8, c16 and c24**, where every decode step changes kernel: base against fix sits in the lower
  part of the reruns' range or a little under it. c8 0.41–0.55 against 0.49–0.97, c16 0.36–0.57
  against 0.45–0.82, c24 0.52–0.67 against 0.60–0.83. That is what two accumulation orders give.
* **Numerics:** the kernel test of §5, with this run's new case at 2 routes an expert, checks the
  run kernel against the host's routed dot.

## 5. Tests

All single-threaded, on this L4, logs in `results/tests/`. Fix tree `/root/r2g/pie` at `2f4f9cf0`;
base tree `/root/src/pie-base` at `1b851247`, without the new case. The test binaries are debug builds,
with the same commands on both trees (`scripts/tests_build.sh`, `tests_moe.sh`, `tests_engine.sh`).

| suite | fix `2f4f9cf0` | base `1b851247` |
|---|---|---|
| kernels-cuda `linear::moe_select` (`kernels-moe-select-{fix,base}.log`) | pending | pending |
| engine-cuda GPU suite `--test engine_cuda`, 60 tests (`engine-cuda-fix.log`) | pending | failures only: pending |

## 6. note-1: the known `moe_select` failure under compute-sanitizer

pending

## 7. New in this run

* **Infrastructure** (status.md has the details):
  * The only environment variable in the agent's Bash shell matching `TOKEN|SECRET|KEY|PASSWORD` is
    `CLAUDE_CODE_MESSAGING_TOKEN` (name only). It was stripped from every build, bench and test
    process.
  * The Hugging Face download worked.
  * sync.sh pushed.
  * note-1 arrived at the first sync.
* **New pod setup for future runs:**
  * The bench venv needs `transformers` for `--pretokenized-prompts`. The first A/B run failed without
    it (`ModuleNotFoundError`, at 19:31Z) and was rerun from the start.
  * A maturin build that starts while the nightly toolchain is still installing fails in
    `crates/builtins/build.rs`. Install `nightly-2026-10-18` with the `wasm32-wasip3` target before
    any build.
  * The `engine-cuda` suite ran much faster than in the previous run (pending: the total time).

## 8. Files

* `scripts/`:
  * `ab.sh`, with `sweep.sh`, `bench.sh` and `make_prompts.py` from the previous run;
  * `tests_build.sh`, `tests_moe.sh`, `tests_engine.sh`, `after_ab.sh`, `sanitize.sh`;
  * analysis: `curve.py`, `summ.py`, `clocks.py`, `agree.py`, `tokeq.py`.
* `results/`:
  * `ab/`: the 24 runs, with `.json`, `.log` and `.clk` for each;
  * `curve-ab.md`, `summ-ab.txt`, `clocks-ab.txt`, `tokens-ab.txt`, `engines.txt`;
  * `tests/`.
