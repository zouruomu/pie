# Outcome: 20261105-2035-a100-sxm-x1-gemma-4-31b-batching

**Verdict: no PR.** A real fix, verified on the target cell, but not shippable as it stands: it reopens the over-admission that #1368's cap prevents, on Gemma-4-26B, the model #1368 was written for.

## Cell
A100 SXM 80 GB × Gemma-4-31B bf16 × batching. Site: pie 465.76 vs vLLM 0.32.0 569.00 (−18.1%, pie@d9c9582b6, #1368's branch head).

## Verified by the orchestrator from raw files
- **Gap on main e4bb499, 3 alternating rounds, spread under 1%:** pie best 474.8 (c256) vs vLLM best 656.5 (c128), −27.7%. Every pie JSON carries the expected engine sha256. Both engines ran at max_model_len 2048.
- **Cause:** pie decodes ~21 sequences against vLLM's 31–35. #1368's cap (`crates/runtime/src/bootstrap.rs:318-340`) admits 5 × 671 × 16 / 2048 = 26, where the pool seats 35. At working_set_percent 50 the pool fit strands ~0.5 GiB: the kv planes keep the coarse map unit (`Pools::fit_the_card`, `crates/engine-cuda/src/store.rs`).
- **Fix 54385f76** (r2g/20261105-2035-a100-sxm-x1-gemma-4-31b-batching, on dc6282d7, 10 files +57/−2): new profile decision `engine.uncapped_admission` (default off), plus A100 `seat_to_end`, `uncapped_admission`, `working_set_percent = 40`.
  - Gemma-4-31B, fix medians: c64 541.3, c128 592.3, c256 588.1, c512 596.6, c1024 598.6 (base 456.7–474.8; vLLM 595.4–656.5). Best vs best: −8.8%.
  - Final rebased A/B: c128 457.7 → 592.5, c512 454.1 → 597.2.
  - pbatch pf32–pf1024 unchanged to +6.8% (1 round). Tokens identical base vs fix where the cap does not bind (raw, c8, c16, pf4–pf512).
  - Tests on 54385f76: runtime 101, pie-profile+worker+engine 78, engine-cuda chip 13, engine-cuda lib 69 pass. Integration: 2 failures (`a_joined_projection_answers_what_its_legs_answer` :245/:309, EP `an_expert_parallel_pair_answers_what_one_rank_answers` :358) reported identical on base e4bb499. 8 more failed on a full disk and passed on rerun. These are taken from the pod's report; I read the first-run logs only.
- **The blocker, Gemma-4-26B-A4B c512, ITL p99 (ms) recomputed from the JSONs:**
  - final fix 1b4c64bf: 89, 680, 430, 456, 90;
  - main a6547509: 360, 177, 190;
  - main e4bb499: 190, 190, 49;
  - vLLM: 137.
  - The stalled fix runs have 2.7–3.1k token gaps over 300 ms (main 0.9–2.0k) and 44–47 sched-stall/suspend log events (main 1–6).
  - The pod's trace: uncapped admission seats all 512 before any process finishes, while the seat-to-end gate has no footprint yet. 17–18 processes are paused to host, and lanes stall in cohorts for 0.6–2.7 s.
  - Throughput (+3–10%) and TTFT p50 stay better in every run. But the tail regression is beyond the base's spread, and pausing lanes is what the cap exists to stop.

## What a follow-up should do
Bound the first wave of uncapped admission until the seat-to-end gate has learned a footprint, for example by keeping #1368's cap until the first process finishes. Then re-measure Gemma-4-26B c512/c1024 (5+ runs each, p99 and pause counts) and Gemma-4-31B c64–c1024. Keep the working-set line only with its honest mechanism (the finer-unit reshape), or fix `fit_the_card` directly. The H100 Gemma-4-31B cell (−24.4%) probably shares the cause.

## Cost and time
Pod 752nvw4kh1pefz, US-WA-1, $1.79/h, created 20:36Z, closed ~11:55Z (≈15.3 h, ≈$27). Early verdict 22:17Z (reproduced −28%). Hand-over 11:38Z (target was 04:36Z): attribution, two re-scopes (notes 1 and 2), and a 3 h verification queue. Review 1 at 10:40Z, answered 11:38Z.

## Infra finding
Notes 2 and 3 never reached the agent. `pod/sync.sh:27-32` marks a note seen when it prints it, and the agent's background queue scripts call sync.sh, so the notes went into queue logs. Only reviews (delivered by bootstrap on session exit) were reliable.
