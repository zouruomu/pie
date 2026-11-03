# Brief: 20261103-2218-l4-x1-gpt-oss-20b-batching

Run id: `20261103-2218-l4-x1-gpt-oss-20b-batching`. Your fix branch is `r2g/20261103-2218-l4-x1-gpt-oss-20b-batching`, and your evidence branch is `r2g-evidence/20261103-2218-l4-x1-gpt-oss-20b-batching`.

## Target cells (NVIDIA L4, 1 GPU, 24 GB, sm_89)

Both cells are from the site's newest snapshot (db updated 2026-11-03T21:38Z). The model is gpt-oss-20b, scheme mxfp4, full model (not a miniature). The best engine is vLLM 0.32.0.

| mode | site metric | pie | vLLM 0.32.0 | SGLang 0.5.22 | gap | pie commit / measured | vLLM measured |
|---|---|---|---|---|---|---|---|
| batching (`throughput` workload) | output_tok_s | 1576.34 | 2192.91 | 508.14 | -28.1% | 8fb7e90e6, 2026-11-02 19:13 | 2026-11-02 18:03 |
| pbatch (`prefill_batch` workload) | prefill_tok_s | 4666.45 | 6424.65 | 558.40 | -27.4% | 8fb7e90e6, 2026-11-02 19:04 | 2026-11-02 17:53 |

The other fields of the site rows (p50/p99):
- **batching:**
  - pie: TTFT 3,993/21,234 ms, ITL 63.1/180.1 ms, latency 40,108/48,370 ms, resident 2.59 GiB.
  - vLLM: TTFT 48,731/98,182 ms, ITL 87.7/333.2 ms, latency 77,914/119,466 ms.

  So pie has better latency and ITL but less throughput. That pattern suggests pie keeps fewer requests in flight (admission, seating, KV pool size or working set), not slower steps. Verify this; don't assume it.
- **pbatch:**
  - pie: TTFT 1,045/1,057 ms, latency 1,046 ms.
  - vLLM: TTFT 3,284/6,043 ms.

  Pie's pbatch prefill_tok_s (4,666) is almost equal to its own raw single-request prefill (4,681 tok/s, TTFT 128.9 ms), and its TTFT is 3× shorter than vLLM's. Before you profile anything, find out from `scripts/bench` and `.github/workflows/evals.yml` (and anton-mel/stats) exactly how the harness computes prefill_tok_s for `prefill_batch`, and which pf-N it reports. The cell may be a harness or metric artifact rather than an engine gap. If so, that is a valid finding. Report it with the evidence and don't "fix" pie for it.
- **raw (green, for reference):**
  - pie: prefill 4,681 tok/s, decode 60.8 tok/s, ITL 16.4 ms.
  - vLLM: prefill 4,773 tok/s, decode 59.8 tok/s.

## Workload definitions (site database `workloads` table)

- `throughput` ("Batching"): concurrency 32, 128 prompt tokens, 128 output tokens. The benchmark set is c32, c64, c256, c8, c4, c16, c128, c512, c1024.
  - The site's single batching number comes from one or more of these concurrencies. Find out which, and reproduce that one first.
  - Report the whole concurrency curve for pie and vLLM.
  - The site's TTFT values (tens of seconds) suggest the high-concurrency end.
- `prefill_batch` ("Batched Prefill"): concurrency 256, 512 prompt tokens, 1 output token. The benchmark set is pf256-512, pf4-512, pf8-512, pf16-512, pf32-512, pf64-512, pf128-512, pf512-512, pf1024-512.
- `raw`: 1 request, 512 prompt tokens, 256 output tokens.

## Current L4 state on main

- **Profile:** `crates/engine-cuda/profiles/sm89/nvidia-l4.toml`, plus the `sm89.toml` tier.
  - It sets `seated_first = true` but not `seat_to_end`.
  - `working_set_percent = 65`, recording bodies at 1024 MB, `forward_tokens = 2304`.
  - `moe.mxfp4_fp8_from = 256`, `moe.mxfp4_mma_per_from = 4`.
- **Changes since the measured commit (8fb7e90e6):**
  - The L4 profile changed only in #1362 (75930c9). Read what it did.
  - Several engine-wide changes landed since, which may already move these cells: #1368 (KV admission, seated first, c57e793), #1401 (MXFP4 bf16 copies only where they fit, 882f246), and the B200/Hopper MXFP4 series.
  - So reproduce on current main first. If the gap is already closed, report "already green".

## Related upstream work (avoid duplicating it; build on it)

Anton Melnychuk is tuning gpt-oss MXFP4 select thresholds and admission per GPU:
- #1355: RTX 4090, gpt-oss decode selects of up to 32 routes an expert on the run kernel.
- #1380: A40, experts of 32 routes or fewer on the register-held MXFP4 select. gpt-oss-20b c256 went 3,599 → 4,132.
- #1375: A40 seating and a 65% working set.
- #1358: RTX 5090, seat_to_end, window_spare, one state slot.
- #1368: A100 admission.

If the L4 cause turns out to be one of those already-known mechanisms, missing from the L4 profile, say so plainly. A profile line justified by L4 A/B numbers is a legitimate fix. If it's something else (engine, kernel or scheduling), find it.

## Related red cells to check on the same pod

- L4 × qwen3.8-flash-next (miniature, 4 of 48 layers, as the site cuts it) × pbatch: pie 25,297.74 vs vLLM 27,884.69 (-9.3%), pie@8fb7e90e6, 2026-11-02 16:20. Check it if the pbatch cause looks shared.
- Not on this pod, but the same model and modes are red on H100 PCIe (pbatch -24.5%, batching -18.2%, pie@44fc984c8, 2026-11-02). If your cause is clearly general rather than L4-specific, say so in the report.
- For regressions, use the L4's other catalog models in batching and pbatch (e.g. qwen3-0.6b, gemma-4-e4b, qwen3.5-0.8b, mimo/qwen3.8 miniatures where they fit). Several of these are currently green on L4: mimo mini batching pie 6,072 vs vLLM 5,117; nemotron-3-super prefill pie 3,074 vs vLLM 2,668. Don't make them red.

## Ledger notes from earlier runs

- **Pilot** (RTX 4090 × Qwen3.5-0.8B decode, pie-project/pie#1356, now merged):
  - Cause: a missing `[compiler] dag_streams` in the 4090 profile.
  - `pie tune` keeps a setting only if it gains ≥1% on its single tuning model (gpt-oss-20b), so settings that matter elsewhere can be missing for that reason alone.
- No earlier run has touched L4 or gpt-oss-20b batching.

## Budget

- One NVIDIA L4 pod. Pod cost at most $60 and at most 16 hours: `R2G_DEADLINE_H` is set on the pod, and the bootstrap terminates the pod at the deadline.
- Aim to hand over well before then: the orchestrator needs time for up to 4 review rounds.
- Push evidence at every phase.

## What the orchestrator expects at hand-over (PLAYBOOK step 7)

- **status.md:** `READY FOR REVIEW`.
- **report.md:**
  - how the site computes each metric;
  - the reproduced baseline on current main, with the commit, SM clocks, ≥3 alternating rounds per engine, the full concurrency curve, and spread;
  - the attribution, with a per-component table that sums to the gap, and `file:line` for pie and vLLM 0.32.0;
  - the fix, line by line, with an A/B for each non-obvious line;
  - before/after tables;
  - full token-id equality (`--dump-all-token-ids`) between base and fix;
  - cross-model, load (c32 through c1024, pbatch, raw TTFT and decode) and memory-fit checks;
  - tests before and after on the same build;
  - the remaining gap, broken down;
  - side findings;
  - playbook notes.
- **pr.md:** title and body, as the PLAYBOOK describes.
- **results/:** every raw JSON and log behind each number, the engine sha256 per run, and the analysis scripts.
- **Freshness:** rebase on the newest upstream main right before hand-over, and rerun the target A/B and tests after the rebase.
- **Already green:** if the gap is gone on current main, or is a harness artifact, write that up with the same rigor and skip the fix.
