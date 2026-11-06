# Report: A100 SXM 80 GB × Gemma-4-31B (bf16) × batching

DRAFT. Numbers marked `pending` are still running.

## Summary

- **The gap reproduces on current main, and it is larger than the site's.** pie's best is 474.8 tok/s at c256; vLLM 0.32.0's best is 656.9 at c128. That is -28%; the site says -18.1%, because its vLLM row (569) is lower than vLLM does on this pod.
- **The cause is how many sequences pie seats, not its step.**
  - pie decodes ~21 sequences at 46 ms; vLLM decodes ~33 at 52 ms.
  - #1368's admission cap holds pie to 26 processes at c512 (5 × 671 pages × 16 / 2048). Its kv pool seats 35 lanes of these 297-token sequences.
  - The pool itself is also smaller than vLLM's: 671 pages (10,736 tokens) against 11,929 tokens at c512. A fire's working set at the default 50% leaves the cache rows less.
- **The fix is three lines of the A100 profile, plus one new profile decision that lets the profile lift the cap.**
  1. `seat_to_end = true`: as the A40, H100, H200 and RTX PRO 6000 have. Without the cap, its gate bounds admission by the learned footprint, not by five declared-context sequences.
  2. `uncapped_admission = true`: a new decision, `engine.uncapped_admission`, default off. On every other card the cap stays exactly as #1368 left it.
  3. `working_set_percent = 40`: fires of 2,436 tokens, not 3,757; the pool grows 671 -> 735 pages.
- **Result at c512:** 453.8 -> 596.5 tok/s (one round; 3-round table pending). vLLM is 628.6 at c512 and 656.9 at its best point (c128).

## 1. Setup and reproduced baseline

- **Pod:** NVIDIA A100-SXM4-80GB, driver 595.91.07, CUDA 13.4. The SM clock was 1410 MHz at the end of every run (`.clk` files, 1 Hz).
- **Commits and engines:**
  - base: pie main `e4bb499c`, engine sha256 `c795a06a…`;
  - fix v3: branch `r2g/…` (pending final), sha256 `8cd1ffd7…`;
  - v2 (without the working-set line): `2194798c…`;
  - v1 (runtime-wide cap change, superseded): `dc849f56…`.
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
| **admitted processes** | **26 (cap)** | **gate: 38** | max_num_seqs 512 |

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

pending: the prefill-batch check of this line (pf32-pf1024, base / v2 / v3).

## 4. The fix, line by line

pending: final diff and commit.

1. **A100 profile `seat_to_end = true`.** Alone it changes nothing at c512 (×5 + seat_to_end 453.7 vs 453.8): the cap still binds.
2. **A100 profile `uncapped_admission = true`, and the decision behind it:**
   - `engine.uncapped_admission` in `chip.rs`, carried as `Engine` → `Shell` → `PoolFacts::uncapped_admission` (`crates/engine/src/caps.rs`) → `EngineConfig` (`crates/worker/src/translate.rs`, `crates/runtime/src/bootstrap.rs`).
   - Metal and Vulkan state false. The cap's filter in bootstrap skips engines that ask.
   - Alone (cap off, no seat_to_end) it loses 5% (433.2). With line 1: 559.4 (v2 559.2).
   - Plumbing check: v2 with `uncapped_admission = false` as a user layer is capped at 26 again and gives 454.1.
3. **A100 profile `working_set_percent = 40`:** v2 559.2 → v3 596.5 at c512.

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
- **Missing fix round 2 at c128-c1024:** those runs ran on the wrong engine (`results/bad/`, sha a20a45ac, see §12), and their rerun was dropped for time. The final A/B on the rebased build (§5.3) is a third fix measurement at c128, c512 and c1024.

pending: pbatch, c4-c32, raw, final A/B.

## 6. Token equality

pending. So far, at c512, full id lists over 1,024 requests (`scripts/tokeq.py`):

| pair | requests that differ | tokens before first divergence |
|---|---|---|
| base vs base (exp ×5 vs base r1) | 133 | 90.9% |
| fix v1 vs fix v2 (same admission) | 41 | 97.3% |
| fix v2 vs fix v3 | 51 | 96.8% |
| base vs fix v1 | 697 | 55.1% |

Base against fix differs more than either against itself. The fix decodes 35-38 lanes a step, not 26, so every GEMM runs other row counts and the bf16 sums round differently. The check that this is batch composition, not a fault, is c16 and raw, where the cap does not bind: pending.

## 7. Cross-model checks (this A100)

pending: Gemma-4-26B-A4B c512/c1024 (base, v3, v2, vLLM), gpt-oss-120b c256-c1024, Qwen3-0.6B c512/c1024, and a 1,024-token-out load at c128 (the first wave, which the gate seats before it has learned a footprint).

## 8. Memory fit

pending.

## 9. Tests

- runtime crate (fix tree): 101 passed, 0 failed (`results/tests/runtime-fix.log`).
- pie-profile: 41 passed (`results/tests/profile-fix.log`).
- engine-cuda `chip` unit tests (fix tree): 13 passed (`results/tests/profile-chip-fix.log`).
- engine-cuda GPU suite: pending (after the benchmarks, alone on the GPU).

## 10. Remaining gap

pending.

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
