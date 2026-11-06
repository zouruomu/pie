# Report: A100 SXM 80 GB × Gemma-4-31B (bf16) × batching


## Summary

- **The gap reproduces on current main, and it is larger than the site's.**
  - pie's best is 474.8 tok/s at c256; vLLM 0.32.0's best is 656.5 at c128 (3-round medians). That is -27.7%.
  - The site says -18.1%; its vLLM row (569 at c1024) is lower than vLLM does on this pod.
- **The cause is how many sequences pie seats, not its step.**
  - pie decodes ~21 sequences at once at 46 ms a token; vLLM decodes 31-35 at 52 ms.
  - #1368's admission cap holds pie to 26 processes at c512 (5 × 671 pages × 16 / 2048). The kv pool seats 35 of these 297-token sequences, so the cap binds first.
  - The pool is also smaller than vLLM's: 671 pages, 10,736 tokens, against 11,929 tokens at c512. At the default working set the pool fit leaves 525-561 MiB of its room unused (two boots, §3.2).
- **The fix: three A100 profile lines, and one new profile decision behind one of them.**
  1. `seat_to_end = true`, as the A40, H100, H200, RTX 4090, RTX 5090 and RTX PRO 6000 profiles have.
  2. `uncapped_admission = true`. The new decision `engine.uncapped_admission` (default off) lets a profile lift #1368's cap so that seat_to_end's gate, which learns the footprint, bounds admission. Every other card keeps the cap exactly as it is.
  3. `working_set_percent = 40`: fires of 2,436 tokens, not 3,757 (about vLLM's 2,496-token chunks). The pool grows 671 -> 735 pages.
- **Result:**
  - Gemma-4-31B c64-c1024: +18.5% to +31.4%. Best point 474.8 -> 598.6 tok/s: -8.8% against vLLM's best, level with vLLM at c1024.
  - c32 +37%; c4-c16 and raw unchanged.
  - pbatch best point 3,735 -> 3,829 (single runs within ±3% elsewhere, -2.6% at pf16).
  - Gemma-4-26B-A4B c512 +10.7%; gpt-oss-120b c1024 +9.1%; Qwen3-0.6B unchanged.
  - With 1,024 tokens out at c128: 148 -> 217 tok/s. The base thrashes there; the fix does not.
- **Costs:** tail latency rises where admission grows.
  - Gemma-4-31B c1024 ITL p99: 49 -> 481 ms (vLLM 649 ms).
  - Gemma-4-26B c512: 3 of 5 runs of 54385f76 suspend lanes, ITL p99 430-680 ms against main's 177-360. With uncapped admission nothing bounds the first wave before a footprint is learned, so 512 are seated where the pool carries 309 to the end; 17-18 processes are paused to host and restored by copy (no replay). Throughput and TTFT stay better than main's (§7).
  - TTFT falls at every target point.
- **Tokens:** identical base vs fix where the cap does not bind (raw, c8, c16, prefill batches; 1 of 8 c4 requests differs). Where it binds, the lanes a step differ, and so do the outputs. Agreement with vLLM's own ids: base 25.9-26.2%, fix 25.2-25.5% of tokens (§6).
- **Remaining gap (-8.8% best against best):** lanes only. vLLM's pool seats 39-43 of these sequences (blocks of 16), pie's 38. pie's step is 3-5% shorter.

## 1. Setup and reproduced baseline

- **Pod:** NVIDIA A100-SXM4-80GB, driver 595.91.07, CUDA 13.4 (`results/site/nvidia-smi.txt`). SM clock, power and memory are sampled at 1 Hz in every run's `.clk`.
  - Under load (> 250 W) the clock drops from 1410 MHz to 1140-1275 MHz for part of every run, both engines: 18-77% of loaded samples over the target runs (c512: base 34%, fix 37%, vLLM 46%), 62-97% over the pf runs.
  - Runs ended at 1230-1410 MHz.
- **Commits and engines:**
  - base: pie main `e4bb499c`, engine sha256 `c795a06a…`;
  - fix v3: the measured fix, built from `fix-wip` 47cfd8b1 on e4bb499, sha256 `8cd1ffd7…`;
  - `d678268d`: v3 squashed and rebased on main `a6547509`, sha256 `a3d5d503…`;
  - **final commit `54385f76`** on main `dc6282d7`, sha256 `1b4c64bf…`. It is v3 plus the review's bootstrap fix (the cap is lifted only while the pools seat to the end, which they do on the A100) and comment text (§4, §5.3);
  - main `a6547509` rebuilt as the final base: `8f115859…`;
  - v2 (v3 without the working-set line): `2194798c…`;
  - v1 (a runtime-wide cap change, superseded after note 1): `dc849f56…`;
  - exp engines (env-set cap multiplier, gate and fit traces; never part of the fix, `exp-src/exp-engine.diff`): `650553b5`, `48ca3e6b`, `a20a45ac`.
- **vLLM:** 0.32.0 in its own venv. It picks TRITON_ATTN (Gemma 4's heterogeneous head dims), chunked prefill at 2,496 tokens and PIECEWISE+FULL CUDA graphs.
- **Workload:** the site's `throughput` workload.
  - Shape: `tput --num-requests 2N --concurrency N --max-tokens 128`, `--max-model-len 2048`, `--gpu-mem-util 0.90`, `--warmup 2` (harness defaults).
  - Prompts: 169 chat-rendered tokens (Gemma's 25-token frame + the 16-token request header + 128). pie runs `--pretokenized-prompts` (ids from the HF chat template); vLLM renders the same template itself. Both count 169 prompt tokens a request (`prompt tokens` in every log).
  - The prompt lengths are inferred from the site's rows: pbatch's 553 = 512 + 41, and the same +41 lands gpt-oss on 225, which the L4 run found.
  - Wrappers: `scripts/bench.sh` and `scripts/sweep.sh`.
- **Which point the site scores** (`results/site/site-rows.txt`, the site's db.js on 2026-11-05):
  - pie's site row (465.76, latency p50 140 s) is c512: 465.76 × 140.4 / 128 ≈ 511 in flight.
  - vLLM's (569.00, latency p99 458 s) is c1024. Its client submits every request at t=0, so the wall is about the latency p99: 569 × 457.6 / 128 ≈ 2,034 requests (an inference, as in the L4 run).
  - On this pod pie's c512 is 453.8 with TTFT p50 132 s and ITL 46.2 ms, the site's row within 3%.

**Round 1 (base and vLLM):**

| tok/s | c64 | c128 | c256 | c512 | c1024 |
|---|---|---|---|---|---|
| pie base | 456.7 | 457.9 | 474.8 | 453.8 | 470.6 |
| vLLM 0.32.0 | 601.2 | 656.9 | 625.0 | 628.6 | 596.8 |

(Full 3-round table in §5.)

## 2. How many sequences each engine seats, and why

**Lanes decoding at once (output tok/s × ITL p50):** pie 21.0 at 46.2 ms; vLLM 30.7-34.8 at 51.4-53.1 ms.

**pie at c512.** From the arm-trace boot (`results/diag/pie-base-e4bb499-c512-armtrace.log`) and the runtime's admission line (`results/exp/x5-c512.log`):

| MiB | pie base | pie fix v3 | vLLM (c512) |
|---|---|---|---|
| device budget (0.9 of 81,153) | 73,037 | 73,037 | 73,042 (0.9 × 79.25 GiB) |
| in use before load (context) | 955 | 955 | 748 (non-torch) |
| weights | 58,918 | 58,918 | 60,406 (vLLM's "model loading took 58.99 GiB"; why it is 1.45 GiB above pie's was not checked) |
| activation arena / peak activation | 1,371 (fires of 3,757 tokens) | 1,249 (2,436) | 1,556 (chunks of 2,496) |
| inputs | 678 | 636 | – |
| graph bodies | 554 | 583 | 471 (outside its budget) |
| kernel scratch | 264 | 208 | – |
| readout rows | 15 | 10 | – |
| guest programs | 64 | 64 | – |
| not itemised by the log (elastic pool alignment, other) | 414 | 252 | – |
| room the pool fit left unused | 525 | 43 | – |
| **kv pool** | **9,280 = 671 pages × 16 = 10,736 tokens** | **10,120 = 735 pages = 11,760 tokens** | **10,332 = 11,929 tokens** |
| sequences the pool seats (19 pages / blocks of 16 each) | 35 | 38 | 39 |
| **admitted processes** | **26 (cap)** | **the gate; its lockstep bound is 735 / 19 = 38 (inferred: the gate was traced at 671 pages only)** | max_num_seqs 512 |

- **Bytes per token:** pie 0.864 MiB, vLLM 0.866 MiB. The kv layouts cost the same, and the window rows hold as many pages as kv pages, since every sequence is under the 1,024-token window.
- **vLLM's pool shrinks with max_num_seqs** (graph capture): 13,307 tokens at c128, 13,125 at c64, 12,328 at c256, 11,929 at c512, 10,889 at c1024. Its best point, c128, has the largest pool.
- **pie's pool also shrinks with the lanes**, at the default working set (exp logs with `RUST_LOG`): 767 pages at c64 (cap 29), 735 at c128 (cap 28), 671 at c512 (cap 26), 703 at c1024.

## 3. Root causes

### 3.1 The five-sequence cap binds before the pool

- **Where:** `crates/runtime/src/bootstrap.rs:318-340` on main e4bb499 (#1368). For a seated-first kv pool it admits `5 × total_pages × page_size / max_context` processes: 26 at c512, 29 at c64.
- **What it assumes:** each sequence holds 2048/5 = 410 tokens. These hold 297 (19 pages), so the pool seats 35 but only 26 are admitted.
- **Why it exists:** without seat_to_end, more admitted prompts than the pool can carry to the end get seated and then suspended for each other's growth.
  - `x0-c512`: cap off, seat_to_end off → 433.2 tok/s with ITL p50 0.09 ms (replays).
- **seat_to_end gates what the cap approximates** (`crates/runtime/src/planner/to_end.rs:66-110`): it holds a process back until the pools carry the started ones to the end, from what finished processes held. A trace of its gate (`results/exp/x0-s2e-trace-c512.log`) shows:
  - it learns 19 pages a finished lane held (11 at the prompt);
  - it admits while `most_at_once` ≤ 671, i.e. up to 35 live lanes, the pool's lockstep bound;
  - oversubscribing it by 10% or 20% (`PIE_TO_END_OVER`, already in main) gives 559.5 and 559.4, against 559.4. The lanes are in lockstep (every request is 128 tokens), so the bound is real.

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
  - The arena, inputs and scratch shrink 220 MiB.
  - Most of the 64 pages come from the pool fit, `Pools::fit_the_card` (`crates/engine-cuda/src/store.rs:1069-1212`). An exp engine printing its inputs (`results/exp/fit-trace-*.log`) shows how:

| fit pass | 50% (base) | 40% |
|---|---|---|
| 1st (before arming, bodies' allowance held) | live 10,975, room 8,673 MiB → 623 pages; the finer-unit test fails (fine ≤ fit; no reshape is printed), the planes keep their unit | live 11,145, room 8,801 → 623 pages; fine 639 > 623, so the planes are re-reserved for reach 799 pages in finer units |
| 2nd (after arming) | room 9,841 → **671** pages; declared(671) = 9,280, declared(672) = **9,880** MiB | room 10,163 → **735** pages; declared(735) = 10,120, declared(736) = 10,320 |

  - At 50% the kv planes (one or two a layer over 60 layers, and the window rows) keep the map unit they were reserved with, for the 9,593-page initial ceiling. Planes of one shape cross a unit at the same page, so one more page costs 600 MiB, and this boot strands 561 MiB. The diag boot in §2 strands 525 of a 9,805 MiB room.
  - At 40% the first pass re-reserves them in finer units: 200 MiB steps, and 43 MiB left.
  - Of the line's +840 MiB of pool, 518 MiB is the stranded room recovered (561 → 43) and 322 MiB is more room (9,841 → 10,163). The underlying issue is §12's first side finding.

| c512, fix v1 + user layer | fire tokens | pool pages | tok/s |
|---|---|---|---|
| working_set_percent 50 (default) | 3,757 | 671 | 559.5 |
| 40 | 2,436 | 735 | 597.0 |
| 30 | 1,449 | 768 | 601.5 |

Prefill batches with and without this line are in §5.2 (v2 is v3 without it).

## 4. The fix, line by line

Commit `54385f76` on main `dc6282d7`: 10 files, +57/-2 (`fix.patch`). It is the measured v3 plus two changes from the self-review, which do not change what the A100 does: the bootstrap filter below, and profile comment text.

1. **`crates/engine-cuda/profiles/sm80/nvidia-a100.toml`: `seat_to_end = true`.**
   - Alone it changes nothing: ×5 + seat_to_end gives 453.7 against 453.8 at c512, because the cap still binds.
   - With line 2 it is what keeps the uncapped pool from suspending lanes.
2. **Same file: `uncapped_admission = true`, and the decision behind it.**
   - `crates/engine-cuda/src/chip.rs`: `UncappedAdmission`, `engine.uncapped_admission`, a flag, default false, registered beside `SeatToEnd`. `Engine.uncapped_admission` is read from the profile; the two test constructors state false.
   - `crates/engine-cuda/src/serve/load.rs`, `serve.rs`: the `Shell` keeps it, with an accessor next to `seat_to_end()`.
   - `crates/engine-cuda/src/api.rs` → `crates/engine/src/caps.rs` (`PoolFacts::uncapped_admission`, `#[serde(default)]`) → `crates/worker/src/translate.rs` → `crates/runtime/src/bootstrap.rs` (`EngineConfig.uncapped_admission`). This is the path `seat_to_end` already takes. Metal and Vulkan state false.
   - `bootstrap.rs:299` and `:330-336`: the runtime-wide `seat_to_end` (every engine asks it) is computed before the cap. The cap's filter then skips an engine that asks, but only while that gate runs: `d.seated_first && !(d.uncapped_admission && seat_to_end) && d.limits.max_context > 0`.
     - Without the `seat_to_end` term, a profile with `uncapped_admission` but no seat_to_end, or a group of engines mixing them, would get neither the cap nor the gate: the x0 case below.
     - v2 and v3 had the filter without it; on the A100 both terms are true.
   - **A/B** (c512):
     - alone, without seat_to_end: 433.2, lanes suspended (ITL p50 0.09 ms);
     - with line 1: 559.4 (exp engine), 559.2 (v2);
     - plumbing check: v2 with a user layer `uncapped_admission = false` is capped at 26 again and gives 454.1.
3. **Same file: `working_set_percent = 40`.**
   - A/B v2 → v3: Gemma-4-31B c512 559.2 → 596.6.
   - On Gemma-4-26B the line changes nothing: its pool is 5,887 pages with and without it (`xrep-*` logs). v2 → v3 there (5066 → 5158 medians) is within the run spread.
   - Prefill batches: v2 → v3 at pf32 is 3,766 → 3,635 (-3.5%), and base → v3 at pf16 is 3,733 → 3,637 (-2.6%), both single runs of 2-5 s. Level or better at pf8 and pf64-pf1024 (§5.2).
   - 30% did not gain more (601.5 at c512) and cuts fires to 1,449 tokens.
4. **Provenance:** `engine.seat_to_end`, `engine.uncapped_admission` and `engine.working_set_percent` are added to the A100's `measured` list.

The bootstrap comment that explains the cap gains two lines: where the pools seat to the end, an engine may ask that gate alone to hold admission.

## 5. Before / after

### 5.1 Target cell

Medians with min-max over rounds. Rounds 2-3 alternate fix, base and vLLM within each round. Round 1 does not: base and vLLM ran on 11-05 20:41-22:15Z, fix r1 at 01:09Z. Sources: (`results/base`, `scripts/table.py base c64,c128,c256,c512,c1024 pie-base-e4bb499 pie-fix-v3 vllm`).

| tok/s | c64 | c128 | c256 | c512 | c1024 |
|---|---|---|---|---|---|
| base e4bb499 (n=3) | 456.7 (456.4-457.2) | 457.9 (457.9-458.5) | 474.8 (474.6-474.8) | 453.8 (453.8-453.9) | 470.6 (470.6-470.7) |
| fix v3 (n=3 at c64, 2 elsewhere) | 541.3 (540.5-541.5) | 592.3 (591.4-593.3) | 588.1 (587.5-588.6) | 596.6 (596.5-596.6) | 598.6 (596.5-600.8) |
| vLLM 0.32.0 (n=3) | 601.8 (601.2-603.0) | 656.5 (656.0-656.9) | 625.0 (623.0-625.6) | 628.3 (626.5-628.6) | 595.4 (595.3-596.8) |
| fix / base | +18.5% | +29.3% | +23.9% | +31.4% | +27.2% |
| base / vLLM | -24.1% | -30.2% | -24.0% | -27.8% | -21.0% |
| fix / vLLM | -10.1% | -9.8% | -5.9% | -5.1% | +0.5% |

- **Best point against best point:** base 474.8 (c256) vs vLLM 656.5 (c128) is -27.7%; fix 598.6 (c1024) vs 656.5 is **-8.8%**.
- **Missing fix round 2 at c128-c1024:** those runs ran on the wrong engine (`results/bad/`, sha a20a45ac, §12), and their rerun was dropped for time. The final A/B (§5.3) is a further fix measurement at c128 and c512.

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

- Best point, the site's metric: base 3,735 (pf4), fix 3,829 (pf8), vLLM here 3,702 (pf256, over pf32-pf1024 only).
  - The site's vLLM pbatch row is 3,793.43 (pf32). Against it the base is 1.5% below and the fix 0.9% above, both single runs inside the ±3% noise.
  - So the fix does not make pbatch worse, but these runs can't call the cell green or red.
- Each pf point up to pf64 is a run of 1-10 s (pf32: 4.7-4.9 s); ±3% is within its noise (base pf8 3,649 vs pf4 3,735). vLLM was run from pf32 up only, so its best is over pf32-pf1024.

**TTFT (target cell, p50, 3-round medians):** c64 8.8 → 1.9 s, c128 23.3 → 17.7 s, c256 60.1 → 42.4 s, c512 131.7 → 99.0 s, c1024 265.3 → 206.9 s (vLLM 7.1 / 21.7 / 43.1 / 101.4 / 213.8 s). pie's client holds the concurrency open, so TTFT includes the wait in pie's admission.

### 5.3 Final pass: the rebased commits against main (`results/final`, 1 run each)

| tok/s | c128 | c512 | engine sha256 |
|---|---|---|---|
| main a6547509 (base) | 457.7 | 454.1 | 8f115859 |
| d678268d (v3 on a6547509) | 592.2 | 595.2 | a3d5d503 |
| **54385f76 (final, on dc6282d7)** | **592.5** | **597.2** | 1b4c64bf |

- The two commits main moved by since a6547509 (#1482 server-web, #1483 swift) touch neither the engine nor this diff.
- 54385f76 differs from d678268d by the bootstrap filter's `seat_to_end` term (true on the A100) and profile comment text.


## 6. Token equality

Full greedy id lists, request by request (`scripts/tokeq.py`; outputs in `results/tokeq-target.txt`, `results/tokeq-curve.txt`).

**Where the cap does not bind, base and fix produce the same tokens, but for one c4 request:**

| shape | base vs fix |
|---|---|
| raw (8 × 256 out) | 0 of 8 differ, 2,048 tokens identical |
| c8, c16 | 0 of 16, 0 of 32 differ |
| c4 | 1 of 8 differs (at token 15). No base-vs-base c4 rerun exists to show this is concurrency jitter. |
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

- **Why base and fix differ more than reruns:** the base decodes 21-22 lanes a step and the fix 27-30, so every GEMM and attention launch runs other row counts. bf16 sums can round differently with the row count and tiling, so near-tied greedy picks flip, and the rest of the sequence follows (the likely mechanism; not traced kernel by kernel). A rerun of one engine keeps the lane count and drifts only with arrival jitter.
- **Fix vs fix is closer than base vs base at c64-c512**, not at c1024 (488 vs 329 differ). The fix's waves are lockstep there (the gate holds lanes to the pool's bound), so its batches repeat run to run.
- **The working-set line barely moves tokens:** at c512, no cap + seat_to_end (`exp/x0-s2e-c512`) vs v3 r1: 51 of 1,024 differ, 96.6% of tokens before divergence.
- **Outside reference:** vLLM 0.32.0's own greedy ids on the same 1,024 c512 prompts, from its offline path, since its streaming client stores none (`results/vllm-tokens`, `results/tokeq-vllm.txt`).
  - Share of tokens before each request's first divergence from vLLM: base 25.9 / 26.2 / 25.9% (r1-r3), fix 25.5 / 25.2% (r1, r3).
  - The fix agrees with vLLM 0.4-1.0 points less than the base does, against a 0.3-point spread across base reruns. Neither is close to vLLM.
  - The reference ran vLLM's offline path at `max_num_batched_tokens=8192`; the serving runs used 2,496.
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
- 26B's pool on this build is 5,887 pages for base, v2 and v3 (`xrep-*` logs, `device_pages=5887`). That is 309 sequences of 19 pages, against the cap's 229 (`seated=229`), so the cap bound there too, more mildly.
- **Tail latency (ITL p99, ms):**
  - base: 190, 190, 49 (c512); 45 (c1024).
  - v2: 266, 92, 684 (c512); 1,040 (c1024).
  - v3: 672, 89, 90 (c512); 919 (c1024).
  - vLLM: 137 (c512); 136 (c1024).
  - **Attributed (Review 1, 11:15-11:40Z):** see "Gemma-4-26B tails" below.

**Gemma-4-26B tails: lanes suspended because the first wave overcommits the pool** (`results/x26b`; scripts `tails.py` to `tails4.py`; runs `rv1-*` and `rv1trace-*`).

*Two kinds of gap over 300 ms* (`tails.py`, `tails2.py`, every 26B run):
- **Token 1 to 2: in every pie run, base and fix alike (750-890 per c512 run).** A request's first token comes out of its prefill, and its second waits while the rest of its admission wave is prefilled; each such gap spans 24-173 other requests' first tokens (median). This is the bounded admission-wave cost. On its own it leaves ITL p99 at 45-92 ms (base r3, every clean fix run).
- **Mid-request (after token 10): these make the p99.** They come in cohorts: 100-260 lanes stop at the same token position (81, 97, 115...) while the other lanes keep decoding at ~1,300-1,800 tok/s.
  - On main, a cohort stalls 353-365 ms (worst 547), once per fire of that length. Main's traced run stalls 36 requests with **0 pauses**, so main's stalls are fire length, not suspension.
  - On the fix, a cohort stalls 0.6-2.7 s.

*Main a6547509 vs 54385f76, alternating, c512* (`rv1-*`; the `rv1trace-*` runs add debug logging and are otherwise identical):

| run | tok/s | TTFT p50 / p99 ms | ITL p50 / p99 ms | latency p99 ms | requests with a mid stall | worst gap ms |
|---|---|---|---|---|---|---|
| main r1 | 4677 | 7112 / 11733 | 30.8 / 360 | 17135 | 858 (83.8%) | 365 |
| main r2 | 4868 | 6552 / 11638 | 30.0 / 177 | 16835 | 134 (13.1%) | 377 |
| main traced | 4682 | 6815 / 12359 | 30.2 / 190 | 17756 | 36 (3.5%) | 547 |
| fix r1 | 5162 | 2263 / 8515 | 27.6 / 89 | 14160 | 0 | 94 |
| fix r2 | 4793 | 3129 / 9707 | 17.0 / 680 | 19285 | 941 (91.9%) | 2687 |
| fix traced r1 | 5155 | 2285 / 8780 | 27.4 / 90 | 14182 | 0 | 95 |
| fix traced r2 | 5073 | 2979 / 8887 | 23.0 / 430 | 17522 | 789 (77.1%) | 2497 |
| fix traced r3 | 5044 | 2843 / 10208 | 18.4 / 456 | 16396 | 937 (91.5%) | 1657 |

- Main's ITL p99 across its 3 runs is 177-360 ms here (earlier e4bb499 runs: 49, 190, 190).
- The fix is bimodal: 2 runs clean (p99 89-90 ms), 3 stalled (430-680 ms). The earlier v2/v3 runs split the same way, 3 clean and 3 stalled.
- With or without the stall, the fix has the higher throughput (4793-5162 against main's 4677-4868) and TTFT p50 2.3-3.1 s against 6.6-7.1 s. Latency p99 is lower on clean fix runs (14.2 s against 16.8-17.8 s), but up to 19.3 s on a stalled one.

*Mechanism* (`tails4.py` on the pie::kv / planner debug logs):
- Main's traced run: pool low point 1,756 free pages of 5,887, no pauses, nothing queued. The cap (229 seated) stays under what the pool carries to the end (5,887 / 19 = 309).
- Clean fix run: low point 325 free, no pauses, nothing queued.
- Stalled fix runs: low point 45 and 6 free pages.
  - 18 and 17 processes are **paused**: KV and window state copied to pinned host memory (`residency copy ... to HostPinned`) and copied back ~0.1-0.5 s later. That is a copy, not a recompute, so no replay of tokens.
  - 61 and 459 report lines show page allocations queued: lanes crossing a page boundary while the pool is empty wait, which is why whole cohorts stop at the same token.
  - Each paused process's own worst gap is 0.5-2.5 s.
  - 4-5 prefill fires walk eagerly (`sealed map holds no body for b8192`), lengthening the stall.
- **Why the fix overcommits:** with `uncapped_admission`, nothing bounds the first wave. seat_to_end's gate only bounds admission once a process has finished and given it a footprint (section 5, the 1,024-token run), so all 512 requests are seated while 512 × 19 = 9,728 end-of-request pages exceed the pool's 5,887.
  - Whether it thrashes depends on how far the first wave's prefills are staggered (earlier finishers free pages for later lanes in time), hence the bimodality.
  - Gemma-4-31B's pool (735 pages) has the same exposure; its ITL p99 at c1024, 49 -> 481 ms (section 5), is probably the same thing. I didn't trace it.
- **So this is the worse case, not only a prefill-burst trade-off:** on Gemma-4-26B c512 the fix suspends lanes in about half its runs. It is bounded (KV copied to host and back, no replay, every request completes, throughput and TTFT still better than main), but ITL p99 reaches 430-680 ms against main's 177-360.
- **A likely remedy, not implemented or measured:** bound the first wave until a footprint is learned. Keep the cap, or seed the gate's footprint from prompt + max_tokens, so admission never exceeds pool / footprint (309 here).

**gpt-oss-120b mxfp4** (`results/xoss120`, 1 round):

| tok/s | c256 | c512 | c1024 |
|---|---|---|---|
| base | 3679.9 | 2822.6 | 2763.0 |
| fix v3 | 3700.4 | 2872.1 | 3013.5 |
| vLLM 0.32.0 | 2430.8 | 2396.9 | 2237.3 |
| fix / base | +0.6% | +1.8% | +9.1% |

One run each. The sweeps don't log the admission line (no `RUST_LOG`), so whether the cap binds for gpt-oss is not shown; the c1024 gain has no traced mechanism.

**Qwen3-0.6B bf16** (`results/xq06`, 1 round; a non-Gemma dense model with a pool far larger than these batches, so the cap is not expected to bind; its admission line was not logged):

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

- The base's cap (28 at c128's 735 pages) admits nearly three times what its pool carries to the end, and lanes are suspended and replayed.
- The fix's gate admits everything until the first process finishes and it has learned a footprint. No thrash shows, as seated_first's contract says (a waiting prompt waits for a lane to finish rather than suspend one): ITL p99 is 47 ms. After the first finish the gate holds ~10 lanes (216.7 × 45.2 ms = 9.8).

## 8. Memory fit

- Every model in this report loads and serves on the fix: Gemma-4-31B, Gemma-4-26B-A4B, gpt-oss-120b, Qwen3-0.6B.
- The fix takes no memory: the working-set line hands 40% of what is past the weights to a fire, not 50%, and the rest to the cache rows.
- The fire budget shrinks with the lanes asked: Gemma-4-31B's fires are 4,871 tokens at c64/c128, 4,096 at c256, 2,436 at c512 and 1,117 at c1024 (base: 6,317 / 6,317 / 5,312 / 3,757 / 2,048).
  - A long prompt is then prefilled in more fires; pf1024 is not slower (3,573 vs 3,494).
- A model that only fit at 50% would now get smaller fires, not a refusal: `tokens_within` halves toward its floor (`serve/load.rs:285`).
- Not checked: a model whose single sequence at the declared context needs the 10% of the card the line moves. None of the four comes near it.

## 9. Tests

Every log names the tree it ran on in its first line, or comes from the times below.

- **On the final commit 54385f76** (`results/tests/final-*.log`, each opening with `tree 54385f76…`), all with 0 failed:
  - runtime: 101 passed;
  - pie-profile + worker + engine: 78 passed;
  - engine-cuda `chip`: 13 passed;
  - `cargo check` of engine-metal and engine-vulkan: clean.
- **engine-cuda suite on 54385f76**: a dev build at opt-level 2, its three test binaries built by `cargo test -p engine-cuda --features cuda` at 09:29Z.
  - The cargo run spent 32 min in one CPU-bound lib test (`every_site_a_retired_rule_fused_folds_into_a_region`, which compiles every catalog deployment under every retired rule) with the GPU idle. 60 lib tests had passed when I stopped it (`engine-cuda-gpu-fix-cargo-partial.log`).
  - I then ran the binaries directly (`scripts/suite.sh`): the lib unit tests on the CPU beside the GPU integration binaries, each `--test-threads=1`. Nothing else ran on the GPU.
  - **lib unit tests: 69 passed, 0 failed** (`suite-lib.log`).
  - **integration binary: 50 passed, 9 failed, 1 ignored** (`suite-integration.log`, 3,442 s).
    - `a_joined_projection_answers_what_its_legs_answer` fails identically on base. The base binary was built from e4bb499 on 11-05 and rerun alone (`rerun-base-a_joined_projection_answers_what_its_legs_answer.log`): qwen35-d0.8b-bf16-kv-bf16 has worst gap 0.22070313 over a spread of 13.1875 on both builds, and both panic at lines 245:5 and 309:5. Pre-existing.
    - The other 8 failed while the pod's disk was full (~10:40Z; two print `No space left on device`, all within one window of thread ids). Rerun on the fix binary once space was freed (the ep binary ran beside the first minute of it): **8 passed, 0 failed** (`rerun8-fix.log`).
    - So the integration binary on 54385f76: 58 passed, 1 ignored, 1 failed, pre-existing.
  - **ep binary: 5 passed, 1 failed** (`suite-ep.log`). `an_expert_parallel_pair_answers_what_one_rank_answers` panics at its line 358:9 on both builds, "the last layer's experts are two selects", a check of the test model's trace. The base e4bb499 binary fails it identically, 5 passed and 1 failed (`rerun-base-ep.log`). Pre-existing.
  - **In sum, nothing fails on the fix that passes on base.**
  - The disk filled because these tests write test checkpoints to /tmp (9.8 GB for one) beside 216 GB of HF blobs. I freed 109 GB of HF blobs no kept model references.
- **Earlier trees, superseded:**
  - runtime 101 passed on v1 (23:34);
  - pie-profile 41 passed and engine-cuda `chip` 13 passed on v2;
  - worker + engine 37 passed, and cargo check of worker, engine, engine-vulkan and engine-metal, on v3 (`results/tests/*-fix.log`).
  - `profile-chip-fix.log` opens with a cargo error for the wrong package name (`profile`; it is `pie-profile`). Its chip half ran.
- **No new test.** The cap is inline in `bootstrap_inner` and has no test on main either. Testing it would need the admission arithmetic factored out of bootstrap; I left that to the reviewers.

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
  - vLLM's is 11,929 tokens: 39 sequences in blocks of 16 (19 blocks each).
  - pie decodes 29.8 of 38 seats on average (78%), vLLM 32.8 of 39 (84%); the rest are prompts being prefilled, and turnover. The token 1→2 gap is p50 459-490 ms in pie (fix r1, r3) and 646 ms in vLLM (`scripts/gaps.py`).
- **At c128, vLLM's best point:** vLLM's pool grows to 13,307 tokens (43 sequences), because it captures fewer graphs at max_num_seqs 128. pie's grows less.
  - Fires are 4,871 tokens there (working set at 40% of a larger room), and the room for bodies and rows is 10,075 MiB (`results/base/pie-fix-v3-c128-r3.log`).
  - pie's lanes stay at ~30 while vLLM's reach 35. The exact c128 pool is not logged in the sweep runs (no arm-trace there).
- **What would close it, not done here:**
  1. The fit's stranded room (§12), which the working-set line only side-steps at some budgets.
  2. Possibly: inputs and parts of the arena scale with `max_forward_requests` = the concurrency (512-1024 lanes), while the pool seats under 40. Inputs are 636-678 MiB at c512; how much of that a smaller lane budget would free was not measured.
  3. Overall footprint at c512: pie's non-weight, non-kv memory is 3.9 GiB with the fix (context 0.93, arena 1.22, inputs 0.62, bodies 0.57, scratch 0.20, programs 0.06, readout 0.01, not itemised 0.25, unfitted 0.04), against vLLM's 2.3 GiB inside its budget (non-torch 0.73 + activations 1.52) plus 0.46 of graphs outside it.

## 11. What carries over to other cards

Nothing changes on another card unless its profile sets `uncapped_admission = true`. The new decision defaults off.

- **H100 SXM**, Gemma-4-31B batching: the site has pie 933.17 vs vLLM 1,235.18 tok/s (-24.4%), measured on b2ec7e880 before #1368 (`results/site/site-rows.txt`).
  - Same 80 GB, same model, and its profile has seated_first and seat_to_end. So #1368's cap applies there now, and it would bind the way it does here if the pool is similar.
  - `uncapped_admission = true` in the H100 profile is the candidate. Its `working_set_percent = 90` leaves the cache rows even less than the A100's default.
  - Not measured; this pod has only the A100.
- **A40, H200, RTX 4090, RTX 5090, RTX PRO 6000** also seat to the end and keep the cap. Each would need its own A/B.
- **Why not lift the cap wherever seat_to_end is on?** v1 did exactly that (`exp-src/fix-v1.diff`, rebuilt from the diff this session printed; v1 was never committed): 559.5 at c512, the same as the profile-scoped v2 (559.2). It was narrowed to a profile decision after the orchestrator's note, so that the change reaches only the card measured here.

## 12. Side findings

- **The pool fit can strand ~0.5 GiB** (`crates/engine-cuda/src/store.rs`, `Pools::fit_the_card`).
  - The first pass re-reserves the kv planes in the finer unit only when that gains pages at its own limit, with the bodies' allowance held. The second pass, after arming, has more room, but the planes keep the coarse unit.
  - Gemma-4-31B on this A100 at the default working set: 600 MiB a page step, 561 MiB of 9,841 stranded.
  - Deciding the reshape at the reach (`live - programs`, which the code already computes for it) should recover it wherever the steps are coarse (not tested). I did not change it: it is engine-wide, and finer units mean more mapping handles while the pool grows on demand. Worth a look by whoever owns the elastic pool.
- **A run slotted into another queue left its engine behind.** The fit-trace runs (q10) ran between q8's points and left `exp-fit.so` (a20a45ac) in the shared engine path. sweep.sh copies its engine only when a round starts, so fix v3's round 2 at c128-c1024 ran on it. The sha256 in each JSON caught it; those files are in `results/bad/`.
- **Eager prefill walks:** "the sealed map holds no body for b… — this shape walks eagerly" appears in 1-4 fires of some runs. It shows on Gemma-4-26B uncapped runs (v2, v3) and on Qwen3-0.6B base and fix, so it predates this change.
- **Gemma-4-26B ITL tails with uncapped admission** (§7): traced to first-wave overcommit and suspension (pause to host, restore by copy) in about half the runs. Remedy proposed (bound the first wave until a footprint is learned), not implemented.
- **`PIE_TO_END_OVER`:** an environment knob in `crates/runtime/src/planner/to_end.rs:101-104` on main. The shell otherwise reads no environment (`no_shell_reads_the_environment.rs`); this is in the runtime, so it is outside that test.
- **Harness:** pie's JSON records the engine sha256 and the GPU clocks at start and end; vLLM's streaming JSON records neither, so the `.clk` files cover them. vLLM's streaming client stores no token ids; its offline path does.
- **The A100 gpt-oss-120b batching cell (site -18.0%, measured on ee9098f60 before #1368) is green on current main:** base 3,679.9 at c256 against vLLM's best here, 2,430.8.

## 13. Playbook notes

- Never edit a shell script in place while a running bash executes it. Rewrite to a new file and `mv`. A mid-run `open('w')` on `bench.sh` made bash resume at its old byte offset and run vLLM under pie's label (`results/bad/`).
- Model downloads: `hf download openai/gpt-oss-120b` brings `original/` and `metal/` too, about 170 GB, and filled the 300 GB disk. Use `--include "*.safetensors" --include "*.json"` or the exclude flags.
- `pgrep -f <name>` inside `kill $(...)` matched my own shell, as in the pilot. Use `ps -eo pid,args | grep "[x]"` and kill PIDs.
- pie's runtime tracing is at warn by default; `RUST_LOG=warn,runtime::bootstrap=info` shows the admission cap's line and the pool's `device_pages`. Set it in every sweep, not only in experiments.
- When a queue is paused to slot other runs in, restore its engine afterwards. Better still, have sweep.sh copy the engine before every point.
- Have a fresh-context reviewer read the draft before the last runs, not after: it found a logic gap in the diff and tests that had run on older trees.
