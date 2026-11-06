# Report: A100 SXM 80 GB × Gemma-4-31B (bf16) × batching

Numbers marked `pending` were still running when this was written.

## Summary

- **The gap reproduces on current main, and it is larger than the site's.**
  - pie's best is 474.8 tok/s at c256; vLLM 0.32.0's best is 656.5 at c128 (3-round medians). That is -27.7%.
  - The site says -18.1%; its vLLM row (569 at c1024) is lower than vLLM does on this pod.
- **The cause is how many sequences pie seats, not its step.**
  - pie decodes ~21 sequences at once at 46 ms a token; vLLM decodes 31-35 at 52 ms.
  - #1368's admission cap holds pie to 26 processes at c512 (5 × 671 pages × 16 / 2048). The kv pool seats 35 of these 297-token sequences, so the cap binds first.
  - The pool is also smaller than vLLM's: 671 pages, 10,736 tokens, against 11,929 tokens at c512. At the default working set the pool fit strands 561 MiB (§3.2).
- **The fix: three A100 profile lines, and one new profile decision behind one of them.**
  1. `seat_to_end = true`, as the A40, H100, H200 and RTX PRO 6000 have.
  2. `uncapped_admission = true`. The new decision `engine.uncapped_admission` (default off) lets a profile lift #1368's cap so that seat_to_end's gate, which learns the footprint, bounds admission. Every other card keeps the cap exactly as it is.
  3. `working_set_percent = 40`: fires of 2,436 tokens, not 3,757 (about vLLM's 2,496-token chunks). The pool grows 671 -> 735 pages.
- **Result:**
  - Gemma-4-31B c64-c1024: +18.5% to +31.4%. Best point 474.8 -> 598.6 tok/s: -8.8% against vLLM's best, level with vLLM at c1024.
  - c32 +37%; c4-c16, raw and pbatch unchanged or better.
  - Gemma-4-26B-A4B c512 +10.7%; gpt-oss-120b c1024 +9.1%; Qwen3-0.6B unchanged.
  - With 1,024 tokens out at c128: 148 -> 217 tok/s. The base thrashes there; the fix does not.
- **Tokens:** identical base vs fix where the cap does not bind (raw, c8, c16, prefill batches). Where it binds, the lanes a step differ, and so do the outputs. Agreement with vLLM's own ids is the same for base and fix (§6).
- **Remaining gap (-8.8% best against best):** lanes only. vLLM's pool seats 40-45 of these sequences, pie's 38. pie's step is 3-5% shorter.

## 1. Setup and reproduced baseline

- **Pod:** NVIDIA A100-SXM4-80GB, driver 595.91.07, CUDA 13.4. The SM clock was 1410 MHz at the end of every run (`.clk` files, 1 Hz).
- **Commits and engines:**
  - base: pie main `e4bb499c`, engine sha256 `c795a06a…`;
  - fix v3: the measured fix, built from `fix-wip` 47cfd8b1 on e4bb499, sha256 `8cd1ffd7…`. Its diff equals the final commit's except for profile comment text;
  - final commit: `d678268d` on main `a6547509`, rebased for hand-over (§5.3), sha256 `a3d5d503…`;
  - main `a6547509` rebuilt as the final base: `8f115859…`;
  - v2 (v3 without the working-set line): `2194798c…`;
  - v1 (a runtime-wide cap change, superseded after note 1): `dc849f56…`;
  - exp engines (env-set cap multiplier, gate and fit traces; never part of the fix, `exp-src/exp-engine.diff`): `650553b5`, `48ca3e6b`, `a20a45ac`.
- **vLLM:** 0.32.0 in its own venv. It picks TRITON_ATTN (Gemma 4's heterogeneous head dims), chunked prefill at 2,496 tokens and PIECEWISE+FULL CUDA graphs.
- **Workload:** the site's `throughput` workload.
  - Shape: `tput --num-requests 2N --concurrency N --max-tokens 128`, `--max-model-len 2048`, `--gpu-mem-util 0.90`, `--warmup 2` (harness defaults).
  - Prompts: 169 chat-rendered tokens (Gemma's 25-token frame + the 16-token request header + 128). pie runs `--pretokenized-prompts`, so both engines prefill identical ids.
  - The prompt lengths are inferred from the site's rows: pbatch's 553 = 512 + 41, and the same +41 lands gpt-oss on 225, which the L4 run found.
  - Wrappers: `scripts/bench.sh` and `scripts/sweep.sh`.
- **Which point the site scores:**
  - pie's site row (465.76, latency p50 140 s) is c512: 465.76 × 140.4 / 128 ≈ 511 in flight.
  - vLLM's (569.00, latency p99 458 s) is c1024: 569 × 457.6 / 128 ≈ 2,034 requests.
  - On this pod pie's c512 is 453.8 with TTFT p50 132 s and ITL 46.2 ms, the site's row within 3%.

**Round 1 (base and vLLM):**

| tok/s | c64 | c128 | c256 | c512 | c1024 |
|---|---|---|---|---|---|
| pie base | 456.1 (x5-c64) | 457.9 | 474.8 | 453.8 | 470.6 |
| vLLM 0.32.0 | 601.2 | 656.9 | 625.0 | 628.6 | 596.8 |

(Full 3-round table in §5.)

## 2. How many sequences each engine seats, and why

**Lanes decoding at once (output tok/s × ITL p50):** pie 21.0 at 46.2 ms; vLLM 30.7-34.8 at 51.4-53.1 ms.

**pie at c512.** From the arm-trace boot (`results/diag/pie-base-e4bb499-c512-armtrace.log`) and the runtime's admission line (`results/exp/x5-c512.log`):

| MiB | pie base | pie fix v3 | vLLM (c512) |
|---|---|---|---|
| device budget (0.9 of 81,153) | 73,037 | 73,037 | 73,042 (0.9 × 79.25 GiB) |
| in use before load (context) | 955 | 955 | 748 (non-torch) |
| weights | 58,918 | 58,918 | 60,406 (with the vision tower) |
| activation arena / peak activation | 1,371 (fires of 3,757 tokens) | 1,249 (2,436) | 1,556 (chunks of 2,496) |
| inputs | 678 | 636 | – |
| graph bodies | 554 | 583 | 471 (outside its budget) |
| kernel scratch | 264 | 208 | – |
| guest programs | 64 | 64 | – |
| room the pool fit left unused | 525 | 43 | – |
| **kv pool** | **9,280 = 671 pages × 16 = 10,736 tokens** | **10,120 = 735 pages = 11,760 tokens** | **10,332 = 11,929 tokens** |
| sequences the pool seats (19 pages each) | 35 | 38 | 40 |
| **admitted processes** | **26 (cap)** | **gate, to the pool's lockstep bound (735 / 19 = 38)** | max_num_seqs 512 |

- **Bytes per token:** pie 0.864 MiB, vLLM 0.866 MiB. The kv layouts cost the same, and the window rows hold as many pages as kv pages, since every sequence is under the 1,024-token window.
- **vLLM's pool shrinks with max_num_seqs** (graph capture): 13,307 tokens at c128, 13,125 at c64, 12,328 at c256, 11,929 at c512, 10,889 at c1024. Its best point, c128, has the largest pool.
- **pie's pool also shrinks with the lanes.** At c64 it is 767 pages, and the cap is then 29.

## 3. Root causes

### 3.1 The five-sequence cap binds before the pool

- **Where:** `crates/runtime/src/bootstrap.rs:318-340` (#1368). For a seated-first kv pool it admits `5 × total_pages × page_size / max_context` processes: 26 at c512, 29 at c64.
- **What it assumes:** each sequence holds 2048/5 = 410 tokens. These hold 297 (19 pages), so the pool seats 35 but only 26 are admitted.
- **Why it exists:** without seat_to_end, more admitted prompts than the pool can carry to the end get seated and then suspended for each other's growth.
  - `x0-c512`: cap off, seat_to_end off → 433.2 tok/s with ITL p50 0.09 ms (replays).
- **seat_to_end does the cap's job exactly** (`crates/runtime/src/planner/to_end.rs:66-110`). A trace of its gate (`results/exp/x0-s2e-trace-c512.log`) shows:
  - it learns 19 pages a finished lane held (11 at the prompt);
  - it admits while `most_at_once` ≤ 671, i.e. up to 35 live lanes, the pool's lockstep bound;
  - oversubscribing it by 10% or 20% (`PIE_TO_END_OVER`, already in main) gives the same 559.4 to the digit. The lanes are in lockstep (every request is 128 tokens), so the bound is real.

| c512, exp engine (cap multiplier from env) | admitted | tok/s | ITL p50 / p99 ms |
|---|---|---|---|
| ×5 (= base) | 26 | 453.9 | 46.2 / 49.4 |
| ×5 + seat_to_end | 26 | 453.7 | 46.2 / 48.8 |
| no cap | 512 | 433.2 | 0.09 / 1127 |
| ×7 | 36 | 548.3 | 49.2 / 317 |
| ×7 + seat_to_end | 36 | 559.5 | 49.3 / 52.3 |
| no cap + seat_to_end | 512 | 559.4 | 49.3 / 51.8 |

### 3.2 A fire's working set takes half of what is past the weights

- **Where:** `crates/engine-cuda/src/chip.rs:72` (default 50), applied at `crates/engine-cuda/src/serve/load.rs:284` (bake: max_forward_tokens 8192 → 3,757 at c512) and `:542` (refit).
- **What 40% changes:** fires are cut to 2,436 tokens, about vLLM's 2,496-token chunk.
  - The arena, inputs and scratch shrink 0.24 GB.
  - Most of the 64 pages come from the pool fit, `Pools::fit_the_card` (`crates/engine-cuda/src/store.rs:1069-1212`). An exp engine printing its inputs (`results/exp/fit-trace-*.log`) shows how:

| fit pass | 50% (base) | 40% |
|---|---|---|
| 1st (before arming, bodies' allowance held) | live 10,975, room 8,673 MiB → 623 pages; the finer-unit test fails (fine = fit), the planes keep their unit | live 11,145, room 8,801 → 623 pages; fine 639 > 623, so the planes are re-reserved for reach 799 pages in finer units |
| 2nd (after arming) | room 9,841 → **671** pages; declared(671) = 9,280, declared(672) = **9,880** MiB | room 10,163 → **735** pages; declared(735) = 10,120, declared(736) = 10,320 |

  - At 50% the ~120 kv planes keep the map unit they were reserved with, for the 9,593-page initial ceiling. All of them cross a unit at the same page, so one more page costs 600 MiB, and the fit strands 561 MiB.
  - At 40% the first pass re-reserves them in finer units: 200 MiB steps, and 43 MiB left.
  - So about 480 MiB of the line's 840 MiB is the reshape firing. The underlying issue is §12's first side finding.

| c512, fix v1 + user layer | fire tokens | pool pages | tok/s |
|---|---|---|---|
| working_set_percent 50 (default) | 3,757 | 671 | 559.5 |
| 40 | 2,436 | 735 | 597.0 |
| 30 | 1,449 | 768 | 601.5 |

Prefill batches with and without this line are in §5.2 (v2 is v3 without it).

## 4. The fix, line by line

Commit `d678268d` on main `a6547509`: 10 files, +51/-1.

1. **`crates/engine-cuda/profiles/sm80/nvidia-a100.toml`: `seat_to_end = true`.**
   - Alone it changes nothing: ×5 + seat_to_end gives 453.7 against 453.8 at c512, because the cap still binds.
   - With line 2 it is what keeps the uncapped pool from suspending lanes.
2. **Same file: `uncapped_admission = true`, and the decision behind it.**
   - `crates/engine-cuda/src/chip.rs`: `UncappedAdmission`, `engine.uncapped_admission`, a flag, default false, registered beside `SeatToEnd`. `Engine.uncapped_admission` is read from the profile; the two test constructors state false.
   - `crates/engine-cuda/src/serve/load.rs`, `serve.rs`: the `Shell` keeps it, with an accessor next to `seat_to_end()`.
   - `crates/engine-cuda/src/api.rs` → `crates/engine/src/caps.rs` (`PoolFacts::uncapped_admission`, `#[serde(default)]`) → `crates/worker/src/translate.rs` → `crates/runtime/src/bootstrap.rs` (`EngineConfig.uncapped_admission`). This is the path `seat_to_end` already takes. Metal and Vulkan state false.
   - `bootstrap.rs:329`: the cap's filter skips an engine that asks: `d.seated_first && !d.uncapped_admission && d.limits.max_context > 0`.
   - **A/B** (c512):
     - alone, without seat_to_end: 433.2, lanes suspended (ITL p50 0.09 ms);
     - with line 1: 559.4 (exp engine), 559.2 (v2);
     - plumbing check: v2 with a user layer `uncapped_admission = false` is capped at 26 again and gives 454.1.
3. **Same file: `working_set_percent = 40`.**
   - A/B v2 → v3: Gemma-4-31B c512 559.2 → 596.6; Gemma-4-26B c512 5066 → 5158 (3-run medians).
   - Prefill batches: v2 → v3 at pf32 is 3,766 → 3,635 (-3.5%), and base → v3 at pf16 is 3,733 → 3,637 (-2.6%), both single runs of about 2 s. Level or better at pf8 and pf64-pf1024 (§5.2).
   - 30% did not gain more (601.5 at c512) and cuts fires to 1,449 tokens.
4. **Provenance:** `engine.seat_to_end`, `engine.uncapped_admission` and `engine.working_set_percent` are added to the A100's `measured` list.

The bootstrap comment that explains the cap is unchanged.

## 5. Before / after

**Target cell.** Alternating rounds (fix, base, vLLM within each round), medians with min-max (`results/base`, `scripts/table.py base c64,c128,c256,c512,c1024 pie-base-e4bb499 pie-fix-v3 vllm`).

| tok/s | c64 | c128 | c256 | c512 | c1024 |
|---|---|---|---|---|---|
| base e4bb499 (n=3) | 456.7 (456.4-457.2) | 457.9 (457.9-458.5) | 474.8 (474.6-474.8) | 453.8 (453.8-453.9) | 470.6 (470.6-470.7) |
| fix v3 (n=3 at c64, 2 elsewhere) | 541.3 (540.5-541.5) | 592.3 (591.4-593.3) | 588.1 (587.5-588.6) | 596.6 (596.5-596.6) | 598.6 (596.5-600.8) |
| vLLM 0.32.0 (n=3) | 601.8 (601.2-603.0) | 656.5 (656.0-656.9) | 625.0 (623.0-625.6) | 628.3 (626.5-628.6) | 595.4 (595.3-596.8) |
| fix / base | +18.5% | +29.3% | +23.9% | +31.4% | +27.2% |
| base / vLLM | -24.1% | -30.2% | -24.0% | -27.8% | -21.0% |
| fix / vLLM | -10.1% | -9.8% | -5.9% | -5.1% | +0.5% |

- **Best point against best point:** base 474.8 (c256) vs vLLM 656.5 (c128) is -27.7%; fix 598.6 (c1024) vs 656.5 is **-8.8%**.
### 5.2 The rest of Gemma-4-31B's curve (1 round, `results/curve`)

| tok/s | c4 | c8 | c16 | c32 | raw (1 × 553 in, 256 out) |
|---|---|---|---|---|---|
| base | 93.8 | 177.5 | 323.8 | 398.6 | 25.3 (ITL 38.94 ms) |
| fix v3 | 93.8 | 177.6 | 323.6 | **546.3** | 25.3 (38.97 ms) |
| fix / base | -0.0% | +0.1% | -0.1% | **+37.0%** | 0.0% |

At c32 the cap already binds. It is 5 × pages × 16 / 2048: 29 at c64's 767 pages, so about 30 at c32's slightly larger pool (not logged), against the 32 asked. Below that nothing changes.

**Prefill batch (Σ prompt tokens / wall, the site's prefill_tok_s; 553-token prompts, 1 out):**

| prefill tok/s | pf4 | pf8 | pf16 | pf32 | pf64 | pf128 | pf256 | pf512 | pf1024 |
|---|---|---|---|---|---|---|---|---|---|
| pie-base-e4bb499 | 3735 | 3649 | 3733 | 3681 | 3705 | 3611 | 3492 | 3485 | 3494 |
| pie-fix-v2 | - | - | - | 3766 | 3704 | 3697 | 3622 | 3657 | 3573 |
| pie-fix-v3 | 3734 | 3829 | 3637 | 3635 | 3790 | 3785 | 3640 | 3722 | 3573 |
| vllm | - | - | - | 3541 | 3678 | 3668 | 3702 | 3692 | 3693 |
| v3 / base | -0.0% | +4.9% | -2.6% | -1.3% | +2.3% | +4.8% | +4.3% | +6.8% | +2.3% |

- Best point, the site's metric: base 3,735 (pf4), fix 3,829 (pf8), vLLM 3,702 (pf256). The pbatch cell stays green.
- Each pf point below pf128 is a run of a few seconds; ±3% is within its noise (base pf8 3,649 vs pf4 3,735).

**TTFT (target cell, p50, 3-round medians):** c64 8.8 → 1.9 s, c128 23.3 → 17.7 s, c256 60.1 → 42.4 s, c512 131.7 → 99.0 s, c1024 265.3 → 206.9 s (vLLM 7.1 / 21.7 / 43.1 / 101.4 / 213.8 s). pie's client holds the concurrency open, so TTFT includes the wait in pie's admission.

### 5.3 Final pass: the rebased commit against main a6547509

pending (`results/final`, c128 and c512).

- **Missing fix round 2 at c128-c1024:** those runs ran on the wrong engine (`results/bad/`, sha a20a45ac, see §12), and their rerun was dropped for time. The final A/B on the rebased build (§5.3) is a third fix measurement at c128, c512 and c1024.


## 6. Token equality

Full greedy id lists, request by request (`scripts/tokeq.py`; outputs in `results/tokeq-target.txt`, `results/tokeq-curve.txt`).

**Where the cap does not bind, base and fix produce the same tokens:**

| shape | base vs fix |
|---|---|
| raw (8 × 256 out) | 0 of 8 differ, 2,048 tokens identical |
| c8, c16 | 0 of 16, 0 of 32 differ |
| c4 | 1 of 8 differs (at token 15) |
| pf4, pf8, pf64, pf512 (first token) | 0 differ, although the working-set line changes the fire sizes |

**Where the cap binds (c32 and up), the lanes a step differ, and so do the outputs.** Base vs fix diverges more than reruns of either engine:

| point | base vs base (r3 vs r2) | fix vs fix (r1 vs r3) | base vs fix (r3) |
|---|---|---|---|
| c64 (128 req.) | 30 differ, 84.0% before divergence | 2, 99.2% | 81, 59.0% |
| c128 (256) | 41, 89.1% | 0, 100% | 168, 62.1% |
| c256 (512) | 66, 90.8% | 6, 99.1% | 361, 55.3% |
| c512 (1024) | 140, 90.4% | 33, 97.7% | 700, 55.3% |
| c1024 (2048) | 329, 88.8% | 488, 86.3% | 1488, 53.3% |
| c32 (64), curve | – | – | 44, 52.3% |

- **Why base and fix differ more than reruns:** the base decodes 21-22 lanes a step and the fix 27-30, so every GEMM and attention launch runs other row counts. bf16 sums round differently with the row count and tiling, so near-tied greedy picks flip, and the rest of the sequence follows. A rerun of one engine keeps the lane count and drifts only with arrival jitter.
- **Fix vs fix is closer than base vs base.** The fix's waves are lockstep (seat_to_end holds lanes to the pool's 35-38), so its batches repeat run to run.
- **Outside reference:** vLLM 0.32.0's own greedy ids on the same 1,024 c512 prompts, from its offline path, since its streaming client stores none (`results/vllm-tokens`, `results/tokeq-vllm.txt`).
  - Share of tokens before each request's first divergence from vLLM: base 25.9 / 26.2 / 25.9% (r1-r3), fix 25.5 / 25.2% (r1, r3).
  - The fix agrees with an independent engine about as well as the base does: 0.5 points lower, against a 0.3-point spread across base reruns.
  - Both engines part from vLLM early. Request 0 diverges at token 44 in every pie run: different kernels, a different rounding.

## 7. Cross-model checks (this A100, same harness)

Prompts are 128 + the 16-token header + each model's chat frame: Gemma 169, gpt-oss 225, Qwen3 165 tokens.

**Gemma-4-26B-A4B bf16, the case #1368 was written for** (`results/x26b`):
- c512 has 3 runs per pie engine: `pie-*-c512-r1` plus `xrep-*-c512-r2/r3`. The repeats were run because v3's first run was low.

| tok/s | c512 runs | c512 median | c1024 (1 run) |
|---|---|---|---|
| base | 4654, 4661, 4879 | 4661 | 4941 |
| v2 (no working-set line) | 5066, 5101, 5009 | 5066 | 5034 |
| v3 (fix) | 4736, 5163, 5158 | 5158 | 5044 |
| vLLM 0.32.0 | 4620 | 4620 | 4682 |

- fix/base: c512 +10.7%, c1024 +2.1%.
- By #1368's own figure, 26B's pool is 5,791 pages: 304 sequences of 19 pages, against the cap's 5 × 5,791 × 16 / 2048 = 226. So the cap bound there too, more mildly. (Its pool on this build was not logged in the sweep runs.)

**gpt-oss-120b mxfp4** (`results/xoss120`, 1 round):

| tok/s | c256 | c512 | c1024 |
|---|---|---|---|
| base | 3679.9 | 2822.6 | 2763.0 |
| fix v3 | 3700.4 | 2872.1 | 3013.5 |
| vLLM 0.32.0 | 2430.8 | 2396.9 | 2237.3 |
| fix / base | +0.6% | +1.8% | +9.1% |

**Qwen3-0.6B bf16** (`results/xq06`, 1 round; a non-Gemma, dense model whose pool never binds):

| tok/s | c512 | c1024 |
|---|---|---|
| base | 27778.6 | 28941.4 |
| fix v3 | 27783.7 | 28802.5 |
| vLLM 0.32.0 | 21237.4 | 19035.3 |
| fix / base | +0.0% | -0.5% |

**First wave, 1,024 tokens out at c128** (`results/long`, Gemma-4-31B; 256 requests, each holding ~75 pages at its end, so the pool seats ~10):

| | tok/s | ITL p50 / p99 / max | TTFT p50 | latency p50 / p99 | wall |
|---|---|---|---|---|---|
| base | 148.1 | 42.1 / 1,162 / 83,922 ms | 291.5 s | 823.1 / 1,202.8 s | 1,769.6 s |
| fix v3 | 216.7 (+46%) | 45.2 / 47.3 / 447 ms | 513.2 s | 559.4 / 606.0 s | 1,209.9 s |

The fix's TTFT is higher: a prompt now waits in the gate for a lane to end, where the base seated it and then suspended it. Request latency is lower (p50 -32%, p99 -50%).

- The base's cap (29) admits nearly three times what its pool carries to the end, and lanes are suspended and replayed.
- The fix's gate admits everything until the first process finishes and it has learned a footprint. No thrash shows, as seated_first's contract says (a waiting prompt waits for a lane to finish rather than suspend one): ITL p99 is 47 ms. After the first finish the gate holds ~10 lanes (216.7 × 45.2 ms = 9.8).

## 8. Memory fit

- Every model in this report loads and serves on the fix: Gemma-4-31B, Gemma-4-26B-A4B, gpt-oss-120b, Qwen3-0.6B.
- The fix takes no memory: the working-set line hands 40% of what is past the weights to a fire, not 50%, and the rest to the cache rows.
- The fire budget shrinks with the lanes asked: Gemma-4-31B's fires are 4,871 tokens at c64/c128, 4,096 at c256, 2,436 at c512 and 1,117 at c1024 (base: 6,317 / 6,317 / 5,312 / 3,757 / 2,048).
  - A long prompt is then prefilled in more fires; pf1024 is not slower (3,573 vs 3,494).
- A model that only fit at 50% would now get smaller fires, not a refusal: `tokens_within` halves toward its floor (`serve/load.rs:285`).
- Not checked: a model whose single sequence at the declared context needs the 10% of the card the line moves. None of the four comes near it.

## 9. Tests

- runtime crate (fix tree): 101 passed, 0 failed (`results/tests/runtime-fix.log`).
- pie-profile: 41 passed (`results/tests/profile-fix.log`).
- engine-cuda `chip` unit tests (fix tree): 13 passed (`results/tests/profile-chip-fix.log`).
- engine-cuda GPU suite: pending (after the benchmarks, alone on the GPU).

## 10. Remaining gap

After the fix, medians of the 3-round table. Lanes decoding = output tok/s × ITL p50:

| point | fix tok/s / lanes / ITL | vLLM tok/s / lanes / ITL | fix / vLLM |
|---|---|---|---|
| c64 | 541.3 / 27.2 / 50.2 ms | 601.8 / 31.8 / 52.9 ms | -10.1% |
| c128 | 592.3 / 29.7 / 50.2 ms | 656.5 / 34.7 / 52.8 ms | -9.8% |
| c256 | 588.1 / 29.4 / 50.1 ms | 625.0 / 32.8 / 52.5 ms | -5.9% |
| c512 | 596.6 / 29.8 / 50.0 ms | 628.3 / 32.8 / 52.2 ms | -5.1% |
| c1024 | 598.6 / 29.9 / 50.0 ms | 595.4 / 30.7 / 51.6 ms | +0.5% |

**All of it is lanes:** pie's step is still 3-5% shorter than vLLM's at every point.

- **At c512, seats:**
  - pie's pool is 735 pages, 38 sequences of 19 pages, and its gate admits to that lockstep bound.
  - vLLM's is 11,929 tokens, 40 sequences.
  - Both decode ~78-82% of their seats on average; the rest are prompts being prefilled and turnover. The token 1→2 gap is p50 488 ms in pie and 646 ms in vLLM (`scripts/gaps.py`).
- **At c128, vLLM's best point:** vLLM's pool grows to 13,307 tokens (45 sequences), because it captures fewer graphs at max_num_seqs 128. pie's grows less.
  - Fires are 4,871 tokens there (working set at 40% of a larger room), and the room for bodies and rows is 10,075 MiB (`results/base/pie-fix-v3-c128-r3.log`).
  - pie's lanes stay at ~30 while vLLM's reach 35. The exact c128 pool is not logged in the sweep runs (no arm-trace there).
- **What would close it, not done here:**
  1. The fit's stranded room (§12), which the working-set line only side-steps at some budgets.
  2. Inputs and the arena are sized for `max_forward_requests` = the concurrency (512-1024 lanes), while the pool seats under 40.
  3. Overall footprint at c512: pie's non-weight, non-kv memory is 3.6 GiB with the fix (context 0.93, arena 1.22, inputs 0.62, bodies 0.57, scratch 0.20, programs 0.06, unfitted 0.04), against vLLM's 2.3 GiB inside its budget (non-torch 0.73 + activations 1.52) plus 0.46 of graphs outside it.

## 11. What carries over to other cards

- The H100 SXM's Gemma-4-31B batching cell (-24.4%, measured before #1368) has the same 80 GB and the same model. Its profile has seated_first and seat_to_end, so #1368's cap now binds there too: 5 × pool / 2048.
- `uncapped_admission = true` should do on the H100 what it does here. Its `working_set_percent = 90` leaves the pool even less than the A100's default does.
- Not measured: this pod has only the A100.

## 12. Side findings

- **The pool fit can strand ~0.5 GiB** (`crates/engine-cuda/src/store.rs`, `Pools::fit_the_card`).
  - The first pass re-reserves the kv planes in the finer unit only when that gains pages at its own limit, with the bodies' allowance held. The second pass, after arming, has more room, but the planes keep the coarse unit.
  - Gemma-4-31B on this A100 at the default working set: 600 MiB a page step, 561 MiB of 9,841 stranded.
  - Deciding the reshape at the reach (`live - programs`, which the code already computes for it) would recover it on every card. I did not change it: it is engine-wide, and finer units mean more mapping handles while the pool grows on demand. Worth a look by whoever owns the elastic pool.
- **`PIE_TO_END_OVER`:** an environment knob in `crates/runtime/src/planner/to_end.rs:101-104` on main. The shell otherwise reads no environment (`no_shell_reads_the_environment.rs`); this is in the runtime, so it is outside that test.
- **pie_bench `--engine-option`:** `build_config` reads it and `common.py` defines it. It works.
- **Harness:** pie's JSON records the engine sha256; vLLM's doesn't record its clocks, so the `.clk` files cover them.

## 13. Playbook notes

- Never edit a shell script in place while a running bash executes it. Rewrite to a new file and `mv`. A mid-run `open('w')` on `bench.sh` made bash resume at its old byte offset and run vLLM under pie's label (`results/bad/`).
- Model downloads: `hf download openai/gpt-oss-120b` brings `original/` and `metal/` too, about 195 GB, and filled the 300 GB disk. Use `--include "*.safetensors" --include "*.json"` or the exclude flags.
- `pgrep -f <name>` inside `kill $(...)` matched my own shell, as in the pilot. Use `ps -eo pid,args | grep "[x]"` and kill PIDs.
- pie's runtime tracing is at warn by default; `RUST_LOG=warn,runtime::bootstrap=info` shows the admission cap's line.
