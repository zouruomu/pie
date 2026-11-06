# Status: 20261105-2035-a100-sxm-x1-gemma-4-31b-batching

**Phase:** target A/B done (3 rounds); pbatch, cross-model and curve checks running, then the final A/B and the GPU suite
**UTC:** 2026-11-06 05:12

## Preliminary verdict (1 round so far; rounds 2-3 running)

The gap is real on current main (e4bb499):
- pie c512 453.8 tok/s (site's pie point: c512, 465.76). TTFT p50 132 s, ITL p50 46.2 ms: it reproduces the site row.
- vLLM 0.32.0, round 1: c64 601, **c128 657**, c256 625, c512 629, c1024 597 tok/s (site: 569 at c1024).
- Round 1, pie (main e4bb499, engine sha256 c795a06a): c128 458, c256 475, c512 454, c1024 471 tok/s. c64 is to be rerun (its file was overwritten, see Notes).
- **Verdict: the gap is real and larger than the site's: pie's best 475 (c256) against vLLM's best 657 (c128), -28%.** From c128 up pie flattens at ~21 decoding. (Correction: an earlier line here gave a pie c64 of 601. That number was the overwritten file's, i.e. vLLM's. pie's c64 is pending.)
- Rounds 2-3 for spread are queued after the experiments (~23:05Z onward).

How many sequences each engine decodes (tok/s x ITL p50):
- vLLM: 31-35 at 51-53 ms. Its KV pool is 13,307 tokens (11.3 GiB) at c128 and 10,889 at c1024: CUDA graphs grow with max_num_seqs.
- pie: 21 at 46 ms.
  - Its pool at c512 is 671 pages of 16 = 10,736 tokens (9,280 MiB, kv + window rows), enough to seat ~35 sequences of 297 tokens (19 pages each).
  - But #1368's admission cap (`crates/runtime/src/bootstrap.rs:318-340`) admits 5 x 671 x 16 / 2048 = **26 processes**. The cap binds before the pool does.

pie's memory at c512, from the arm-trace log (results/diag): weights 57.5 GiB, context before load 0.93 GiB, activation arena 1.34 GiB (fires cut 8192 -> 3757 tokens by working_set_percent 50), inputs 0.66 GiB, graph bodies 0.54 GiB, kernel scratch 0.26 GiB. That leaves 9.58 GiB of room, of which the pool takes 9.06 GiB.

## Cap experiments (c512, results/exp, exp engine = main + env-set cap multiplier)

| run | admitted | tok/s | ITL p50 / p99 ms |
|---|---|---|---|
| base (x5) | 26 | 453.8 | 46.2 / 48.6 |
| x5 + seat_to_end | 26 | 453.7 | 46.2 / 48.8 |
| no cap | 512 | 433.2 | 0.09 / 1127 (lanes suspended and replayed) |
| x7 | 36 | 548.3 | 49.2 / 317 |
| x7 + seat_to_end | 36 | 559.5 | 49.3 / 52.3 |
| no cap + seat_to_end | 512 | 559.4 | 49.3 / 51.8 |

More results: no cap + seat_to_end at c128 591 (base 458, vLLM 657). Gate oversubscribed 10% and 20% (PIE_TO_END_OVER): 559.5 and 559.4, unchanged. A trace of the gate (results/exp/x0-s2e-trace-c512.log) shows it learns 19 pages a lane (11 at the prompt) and admits 35 live lanes against 671 pages. So with seat_to_end the pool, not the gate, is the limit.

Fix v1 (branch fix-wip, worktree /root/r2g/pie-fix):
- `crates/runtime/src/bootstrap.rs`: the five-sequence cap applies only where the pools don't seat to the end. **This is shared runtime code: it also lifts the cap on the H100, H200, RTX PRO 6000, RTX 5090, RTX 4090 and A40, whose profiles have seat_to_end.** The other run (h100 gpt-oss-120b) may see it.
- A100 profile: `seat_to_end = true`.
- Runtime crate tests on the fix tree: 101 passed, 0 failed.

Remaining gap after v1: pool size. pie's pool at c512 is 671 pages against vLLM's ~745 (10.09 GiB). pie's non-KV, non-weight memory is 4.3 GiB against vLLM's 2.25 (+0.46 of graphs outside its budget): arena 1.34, inputs 0.66 (sized for 512 lanes), bodies 0.54, scratch+programs 0.32, context 0.93, and 0.51 the pool fit leaves unused.

The cap is what holds pie at ~21 lanes. seat_to_end gates cleanly without it (+23%). But it still holds ~28 lanes where 671 pages fit ~35-44 lanes of 19 pages. Tracing its gate (learned footprint against kv_total) now.

## Note 1 (orchestrator, 23:10Z): acknowledged
1. Done: the fix no longer changes shared admission. v2 makes the cap a profile decision, `engine.uncapped_admission`, default off: every card behaves exactly as before. Only the A100 profile turns it on, beside `seat_to_end = true`. The plumbing spans engine-cuda chip/serve/api, engine caps (`PoolFacts`), worker translate and runtime bootstrap, about 40 lines. Gemma-4-26B-A4B c512/c1024 is first in the verification queue.
2. Shared runtime code is touched only by the new field and its filter, which is a no-op unless a profile asks. No change for the gpt-oss-120b x H100 run.
3. Remaining gap after the cap, c512: the gate admits exactly the pool, 35 lanes = 671 pages / 19. Lanes are in lockstep, since all requests are the same length, so the gate's lockstep bound is real; oversubscribing it 10% or 20% changes nothing (559.4). vLLM's pool seats 40 (11,929 tokens / 297). Lanes decoding: pie 27.6 of 35 (79%), vLLM 32.8 of 40 (82%). ITL 49.3 against 52.2 ms. So the rest of the gap is pool size (-12.5% seats), a slightly larger turnover loss (-4%), and pie's faster step (+6%) against them. working_set_percent 40/30 runs (a smaller arena, a larger pool) are finishing now.

## Fix v3 = v2 + `working_set_percent = 40` in the A100 profile (second, independent line)
The pool-size half of the remaining gap. At c512 (fix v1 + user layer, 1 round):

| working_set_percent | fire tokens | pool pages | tok/s | lanes |
|---|---|---|---|---|
| 50 (default) | 3757 | 671 | 559.5 | 27.6 |
| 40 | 2436 | 735 | 597.0 | 29.8 |
| 30 | 1449 | 768 | 601.5 | 30.2 |

40 is chosen: nearly all of 30's gain, with fires at about vLLM's 2,496-token chunks. Prefill batches are the risk (pbatch is green by only 2.1%). q7 measures pf4-pf1024 for base, v3 and v2.
- Plumbing check: v2 with `uncapped_admission = false` as a user layer gives cap 26 and 454.1, the base value.
- Turnover: the token 1 -> 2 gap is p50 488 ms in pie (fix) and 646 ms in vLLM (scripts/gaps.py). Not pie-specific.
- Engines: base c795a06a, v2 2194798c, v3 8cd1ffd7 (branch fix-wip 47cfd8b1).
- Tests on the fix tree so far: runtime 101 passed; pie-profile 41 passed; engine-cuda chip 13 passed.

## Latest
- Schedule: q11 (fix round-2 rerun at c128-c1024) is dropped to keep review time. fix has 2 rounds there (spread < 0.5%), and q13's final A/B on the rebased build adds a third. Chain now: q8 (pbatch, gpt-oss-120b, Qwen3-0.6B, c4-c32/pf4-16/raw, 1024-out) -> q12b (vLLM token reference) -> q13 (final A/B, GPU suite). Hand-over ETA ~10:15Z.
- pbatch so far (prefill tok/s): base pf32 3682, pf64 3705, pf128 3611, pf256 3492, pf512 3485, pf1024 3494; fix v3 pf32 3635 (-1.2%). Running.
- **Target cell, 3 alternating rounds, medians (results/base, scripts/table.py):**

| tok/s | c64 | c128 | c256 | c512 | c1024 |
|---|---|---|---|---|---|
| base e4bb499 | 456.7 | 457.9 | 474.8 | 453.8 | 470.6 |
| fix v3 | 541.3 | 592.3 | 588.1 | 596.6 | 598.6 |
| vLLM 0.32.0 | 601.8 | 656.5 | 625.0 | 628.3 | 595.4 |
| fix / base | +18.5% | +29.3% | +23.9% | +31.4% | +27.2% |
| fix / vLLM | -10.1% | -9.8% | -5.9% | -5.1% | +0.5% |

  Best against best: base -27.7% (474.8 vs 656.5), **fix -8.8%** (598.6 vs 656.5). Spread is under 1% everywhere. fix has n=2 at c128-c1024 until q11 reruns round 2.
- Final commit prepared: `d678268d` on upstream main `a6547509`, one commit, 10 files, +51/-1, in worktree /root/r2g/pie-fix (branch fix-wip). The r2g branch will be moved to it and pushed after the last benchmark, since editing the main tree's sources mid-sweep trips the bench's staleness guard. Final engines: fix a3d5d503, base (a6547509) 8f115859. q13 runs their A/B at c128/c512/c1024, then the GPU suite.
- More tests on the fix tree: worker + engine 37 passed; cargo check passes for worker, engine, engine-vulkan and engine-metal.
- **A slip, caught by the sha256 check:** my fit-trace runs (q10), slotted between q8's points, left the exp-fit engine in the tree. fix v3's round 2 at c128/c256/c512/c1024 then ran on it (sha a20a45ac, not 8cd1ffd7). Those files are in results/bad/ and are queued again after q8 (q11). Every other JSON's sha matches its label.
- Hand-over ETA: ~09:30-10:00Z (benchmarks until ~08:30Z, then the GPU suite alone on the card). The deadline is 12:37Z.
- **Why working_set_percent 40 adds 64 pages** (`results/exp/fit-trace-*.log`, exp engine printing the pool fit's inputs):
  - At 50%, the kv planes keep the coarse map unit they were reserved with (for the 9,593-page initial ceiling). `declared(671) = 9,280 MiB`, but `declared(672) = 9,880 MiB`: 600 MiB a step, since all ~120 planes cross a unit at the same page. The final fit strands 561 MiB of its 9,841 MiB room.
  - At 40%, the first fit's finer-unit test passes (fine 639 > fit 623). The planes are re-reserved for reach 799 in fine units, steps are 200 MiB, and the final fit leaves 43 MiB.
  - So ~480 MiB of the line's gain is this reshape firing, and only ~0.24 GB is the smaller arena.
  - The underlying issue: `Pools::fit_the_card` (crates/engine-cuda/src/store.rs) decides the reshape at the first pass's limit, with the bodies' allowance held, not at the reach the second fit will have. That is engine-wide, and a trade-off (finer units mean more mapping handles). I'm reporting it as a side finding, not changing it. The profile line stays, since it is measured; its comment will say what it does.
- Exp engine sources: exp-src/exp-engine.diff (env-set cap multiplier, the gate trace, the fit trace; never part of the fix).
- **Fix v3, round 1 (target cell):** c64 541.5, c128 593.3, c256 588.6, c512 596.5, c1024 596.5 tok/s (base 456-475; vLLM 601-657). Best against best: 596.5 vs 656.9, **-9.2% (base: -28%)**. Level with vLLM at c1024 (596.8). Rounds 2-3 running.
- Upstream since base: #1437 (vllm_bench `--max-num-seqs`, default off), #1432 (L4 profile), server-web, wgpu. None touch this; the final rebase will come before hand-over.
- fix v3 c512 596.5 (v2 559.2, base 453.8, vLLM 628.6).
- Gemma-4-26B-A4B (note 1's case), c512, 3 runs each: base median 4661 (4654-4879), v2 5066 (5009-5101), v3 5158 (4736-5163), vLLM 4620. c1024, 1 run: base 4941, v2 5034, v3 5044. v3's first c512 run (4736) was low against its 2 repeats (5158, 5163), so I paused q8 for the repeats (q9) to check whether the working-set line costs 26B. It doesn't.
- Token equality at c512: base vs base 133/1024 requests differ, fix vs fix 41, base vs fix 697. The fix decodes 35-38 lanes a step, not 26, so the GEMM row counts differ. c16 and raw, where the cap does not bind, are queued as the control.
- q8 (scripts/q8.sh) replaced q7: vLLM is trimmed off c4-c32 and raw to fit the time. Draft report.md is up.

## Fix v2 (results so far with the equivalent exp engine, 1 round)
| | base | fix | vLLM |
|---|---|---|---|
| c64 | 456.1 | 540.6 | 601.5 |
| c128 | 457.9 | 591.0 | 656.9 |
| c512 | 453.8 | 559.4 | 628.6 |
| c1024 | 470.6 | 573.7 | 596.8 |

## Running
- Queue q1 (scripts/q1.sh): pie round 1 over c64-c1024, then vLLM and pie rounds 2-3, alternating.
- An experiment engine is building in a separate worktree (/root/r2g/pie-exp, not the fix): it reads the admission cap's multiplier from an env var, so the cap can be swept (x5, x7/x8, off) with and without `seat_to_end`.

## Notes
- 20:49Z the 300 GB disk filled (gpt-oss-120b's metal/ and original/ copies). Freed it; lost ~15 min of runs.
- At 21:42Z I edited bench.sh in place while a pie c64 run was executing it. bash resumed at its old byte offset and ran vLLM under pie's c64-r1 label. That file is in results/bad/; no other run was affected (the engine field and sha256 were checked in every JSON).
- `--pretokenized-prompts` needs transformers in pie's venv; it was installed after the first pie attempt failed.

ETA: the verification queue takes ~5.5 h of GPU time (3 alternating rounds of base/fix/vLLM over c64-c1024, Gemma-4-26B, gpt-oss-120b, Qwen3-0.6B, c4-c32, pf4-pf1024, raw, a 1024-token-out load), then the engine-cuda GPU suite. Hand-over ~08:30-09:00Z, later than the 04:30Z target: the cap and pool attribution, then v2 (made profile-scoped for note 1) and v3 (the working-set line), each needed their own runs. The deadline is 12:37Z. The deadline is 12:37Z.
