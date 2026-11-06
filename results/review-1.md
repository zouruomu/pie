# Review 1 of report.md (independent, read-only)

Method: re-ran `pbtable.py baseline` (output identical to baseline-table.md and to the report's tables), re-ran `kern_table.py` on the four pf256 equal-token CSVs, dumped every pie/vLLM/TRT JSON in baseline/ explore/ equal-tokens/ prof/ batching/ clocks/ smoke/, read the W4A8 nsys sqlite per launch, segmented clocks/*.smi.csv by phase, and opened every pie and TRT-LLM file:line cited.

Everything not listed below checked out: baseline cells, medians, spreads, n, % gaps, TTFTs; component table (all cells within 0.5 ms); the 97% figures (96.8% / 96.9%); 1.9x / 2.1x; 7.17 GFLOP/token, ~517 and ~270 TFLOP/s; the +7% / +18% / 16% component ratios; the W4A8 leg medians (1.4115 / 0.7401 ms) and their TFLOP/s; the 37.9k / 39.4k / 41.9k arithmetic; 354 MB; 229 GB; ee9098 28.7k; vLLM MARLIN and FLASHINFER (in all 26 vLLM logs); the W4A16 refusal log; the loader refusal text; and all pie and TRT file:line references except the ones noted below.

## Findings

### Wrong

1. **The pf32 range is misstated.** Report (Summary, l.8): "The gap is the same −33…−35% from pf32 to pf1024 on both models."
   - Raw data: pf32 is −30.6% (20b) and −28.5% (120b). pf64 is −32.5% / −32.6%. Only pf128 and up fall in −32.8…−35.3%.
   - Fix: "−33…−35% from pf128 to pf1024 (−29…−31% at pf32, −33% at pf64)."

2. **The small-point gaps are stale round-1 numbers.** Report (l.8): "At pf4/pf8, pie … trails by 7–9% on 20b."
   - Table: −9.3% (pf4) and −10.0% (pf8, median of 4). The 7% comes from round 1's pf8 (41.3k), as in status.md.
   - Fix: "trails by 9–10% on 20b."

3. **The 120b best point is a single round, not a median.** Report (l.8): "120b runs 29.1k vs 43.4k (−33%)."
   - pie's 120b medians top out at 29.0k (pf512 28,971; pf1024 29,024). 29.1k is one round's max.
   - Fix: "29.0k vs 43.4k (−33%)", which is consistent with the table.

4. **The spread claim fails at pf256.** Report (l.17): "pf256 +0.2%, pf1024 −0.8%, 3 alternating rounds, spreads <1.5%."
   - pie 120b pf256 is n=5, 28,475–29,017, a 1.9% spread. pf1024 is n=5 as well. Only the vLLM side has n=3.
   - Fix: "pie n=5 / vLLM n=3, spreads ≤1.9%."

5. **pie's power during the measured window is 600 W, not 561 W.** Report (l.32): "pie 120b pf1024 in-run … SM median 2400 MHz (2295–2415), 561 W median." Report (l.91): "pie draws less at the same points."
   - Segmenting clocks/pie-120b-pf1024.smi.csv by phase: the ≥90%-util samples include ~5 s of pre-measurement warmup at 420–550 W.
   - The measured window is the contiguous 156-sample block from 12:27:58.9 to 12:28:14.4, i.e. 15.5 s against a wall of 15.79 s. In it, SM median is 2400 MHz (min 2392) and **power median is 600 W, at the cap**.
   - So pie does not draw less: both engines sit at the 600 W cap, and pie holds ~2400 MHz against TRT's ~2295–2317 MHz.
   - The `.clocks` files are taken before and after the process (~92–98 W afterwards), so they say nothing about in-run power.
   - Fix: "measured window: 2400 MHz median, 600 W (at the cap)". Drop "pie draws less". The conclusion that clocks are not the gap stands, and is stronger.

6. **The effort total does not add up.** Report (l.211): "Total: ~13–19 h … Without the fused epilogue … ~11–15 h."
   - Items 1–5 sum to 4–6 + 4–6 + 1–2 + 3 + 3–4 = **15–21 h** with the fused epilogue.
   - With item 2 unfused (2 h) they sum to **13–17 h**.
   - Fix the totals, or the item estimates.

7. **The 20b "projected win" does not use the rate it states.** Report (l.171–173): "Suppose pie's expert GEMMs ran at TRT-LLM's measured ~517 TFLOP/s … 20b: 2091 → ~1000 ms, so 2.84 s → ~1.76 s, ~74k vs TRT-LLM's 72k."
   - 20b expert work is 4.78 GFLOP/token (24 layers, top-4) × 131,072 tokens = 6.26e14 FLOP. At 517 TFLOP/s that is **~1,211 ms**, not ~1,000.
   - ~1,000 ms is TRT's own 20b time (991 ms), which is ~632 TFLOP/s. The report never states that 20b rate.
   - At 517 TFLOP/s, 20b becomes 2.84 − 0.88 = 1.96 s, which is **~67k (below TRT's 72k)**.
   - Fix: state which rate is used, and give both results.

### Overstated

8. **The headline projection is contradicted by the report's own measured projection.** Report (l.172–174): "120b would reach ~44k tok/s, level with TRT-LLM … So the precision alone closes both cells." Also the Summary (l.8): "the whole gap is the MoE's precision".
   - The same section's measured projection, using pie's own W4A8 kernel, gives **37.9k (padded K, unfused) to 41.9k (true K, fused)**. Against TRT's 43.2k at pf256 that is −12% to −3%.
   - The ~44k path assumes a 10 ms quantize (TRT's `quantize_with_block_size`). pie's measured `quantize_rows_mxfp8_routed` costs ~0.154 ms per layer-fire (0.040 gate-up input + 0.114 down input, per-launch medians). That is ~0.14 s over 921.6 layer-fires.
   - That path also ignores the down-input quantize, which TRT fuses into `doActivationKernel`.
   - 20b has no measured projection. Applying the measured 120b W4A8 per-layer-fire cost (2.17–2.31 ms) to 20b's 614.4 layer-fires (W4A16 3.40 ms each) gives roughly **59–63k vs 72k**. Caveat: the kernel was timed at 128 experts, not 32.
   - Fix: lead with the measured projection, say "closes most of the 120b gap; 20b unmeasured, likely not closed", and soften "whole gap is the MoE's precision" to "the gap is the MoE expert GEMMs, which TRT runs at W4A8".

9. **"Kernel deltas sum to the wall delta" holds for 120b only.** Report (l.124): "pie's kernel time is within 0.5% of its wall time … and the kernel deltas sum to the wall delta."
   - 120b: pie kernel 4582.4 vs wall 4604 (0.47%); deltas +1542 vs +1525.
   - 20b: pie 2821.8 vs 2840 (0.64%, over 0.5%). TRT has 78 ms of idle (1790 kernel vs 1868 wall). The kernel delta is +1031 against a wall delta of +972, a 59 ms mismatch.
   - Fix: "within 0.5% (120b) / 0.7% (20b); on 20b TRT-LLM has 4% idle, so its kernel delta overstates the wall delta by ~60 ms."

10. **The site's 120b shortfall is not explained by the site run alone.** Report (l.18, l.229): "The site's 13.8k for that commit was that site run", and "ran low for both engines."
    - On this pod: site vLLM 22.3k vs 29.3k here = 0.76×. Site pie 13.8k vs 28.7k (ee9098) = 0.48×.
    - The site-day effect is not uniform: pie's site number has a further ~1.6× unexplained beyond vLLM's shortfall.
    - The evidence shows ee9098f60 is fast on this pod (n=1 bench plus 1 nsys run). It does not show what happened on the site.
    - Fix: "ee9098f60 runs 28.7k here; the site's 13.8k is not reproducible on this pod (vLLM also ran 24% low that day, pie 52% low)."

11. **The ea800efeb check does not cover the large points.** Report (l.71): "ea800efeb … is not faster than main at pf8–pf32 … so main has not regressed."
    - Only pf8/16/32 were run, n=1 each, at points with 30–41k round-to-round noise.
    - ea800e: 34.3 / 40.8 / 43.1k. main explore: 36.1 / 43.1 / 43.8k.
    - The report does not say which pf the site's 49.8k is at. If it is a large point (where main's best is 47.0k), pf8–32 does not test it.
    - Fix: name the site's pf. Say "no regression at pf8–32 (n=1)", or run ea800efeb at the site's pf.

12. **The batching comparison is n=1 with mismatched latency metrics.** Report (l.224): "In this shape pie leads by 15%, so the site's −30% does not come from the MoE precision at large batch."
    - One run each, in a decode-heavy shape (128 in / 128 out). pie's 9.98 s is end-to-end latency p50, not TTFT, beside TRT's TTFT p50 of 1.85 s.
    - The report itself says the site's request shape is unknown, so the 15% lead cannot attribute or exonerate the site's −30%.
    - The "#1368's cap … is lifting" clause is garbled. #1368 (c57e7938) is already on main.
    - Fix: "In this shape (n=1) pie leads by 15%; the site's −30% is unexplained", and drop or clarify the #1368 sentence.

13. **The equal-token numbers come from the profiled runs.** Report (l.94): "120b pf256 28.5k (vs 28.9k at 448), 20b 46.1k (vs 46.7k). Results in results/equal-tokens and results/prof/*-t512."
    - 28.5k and 46.1k are the **nsys-profiled** runs (prof/*-t512: 28,471 and 46,149).
    - The clean equal-token runs (equal-tokens/) are 28,980 and 46,635: +0.3% and −0.1% against the 448-token medians.
    - Fix: cite the clean runs, and mark the prof numbers as under nsys. status.md's "pie 28.5k vs vLLM 28.9k (−1.4%) at equal tokens" also uses the profiled number; the clean figure is +0.3%.

14. **The dense rate is more like ~350 TFLOP/s than ~400.** Report (l.145): "cuBLAS's bf16 dense GEMMs run ~400 TFLOP/s in the same trace."
    - Dense work is ~1.94 GFLOP/token × 131,072 tokens = 2.54e14 FLOP over 727 ms, which is **~350 TFLOP/s overall**.
    - By kernel: the qkv-sized nvjet 96x256 is ~360, the o-sized 144x192 is ~390.
    - The argument (a perfect W4A16 select cannot reach 517) still holds against the ~500 bf16 peak.
    - Fix: "~350–390".

15. **The "only path" claim needs scoping.** Report (l.10): "This is its only gpt-oss MoE path on sm_120."
    - Supported by source. CUTLASS W4A16 MXFP4 is sm90-only, and Triton's `TritonFusedMoE` is SM90-only (fused_moe_triton.py:1611). The MARLIN and b12x impls are NVFP4.
    - But the forced run only tried `requested CUTLASS`.
    - Fix: cite the TRITON SM90 check alongside the CUTLASS one. Otherwise OK.

16. **The glue saving has more than one cause.** Report (l.13): "pie's MoE glue is 168 ms cheaper, because its GLU and bias fuse into the select."
    - The saving is also TRT's permute pass (`expandInputRowsKernel`, 61 ms) and TRT's activation quantizes, which pie does not need at W4A16.
    - pie still runs a separate `moe_bias_sum` (51 ms on 120b), so only the gate-up bias is fused.
    - Fix: mirror l.139, which already says this correctly.

### Unclear

17. **Which reproduction points are meant.** Report (l.87): "Reproduces the site: 20b 71.5k (site 71.3k), 120b 43.3k (site 42.4k)."
    - 71.5k matches 20b pf256. No 120b median is 43.3k: pf256 is 43.2k and pf512/1024 are 43.4k.
    - State the pf, and that 120b reproduces +2% above the site.

18. **Prompt shape for the site's batching run.** Report (l.232): "on the 128-token batching shape, pie 64 vs 128."
    - This conflicts with l.224 ("the site's batching request shape is not known to me").
    - Here, pie got 128 of its own tokens (prompts-gptoss-192.json, 131,072 / 1024).
    - Say this is a conditional: if the site feeds a 128-HF-token file, pie prefills 64.

19. **The pf256 projection uses the wrong TRT-LLM comparator and wall.** Report (l.185–190), "vs TRT-LLM's 43.4k":
    - The projection is at pf256, where TRT is 43.2k.
    - The base wall (4.604 s) is the profiled run. The clean run is 4.523 s, which would raise every projection ~0.5–1k.

20. **The quantize row in the W4A8 table mixes statistics.** Report (l.183): "`quantize_rows_mxfp8_routed<32>` 0.081 ms a call", under a column headed "median of 5".
    - 0.081 is the **mean of all 10 launches across both legs**. The per-leg medians are 0.040 ms (gate-up input) and 0.114 ms (down input).
    - The sum is about the same, but label it.
    - Also, the "true K" column is padded-time useful FLOP/s, not a measurement at K=2880. 481.4 rounds to 481, not 482.

21. **Which pie deployment ran.** Report (l.152): "pie serves it as stored (deployments at models/gpt_oss.poem:17-31)."
    - The cited block also declares u4g64/u4g32 dense-quantized and e4m3-KV deployments; the first-listed 20b deployment is u4g64 with a dflash drafter.
    - The run used `gptoss-*-bf16-mxfp4-kv-bf16` (named in explore/w4a8dep-20b-pf256.log; bf16 nvjet kernels in the traces).
    - Name that deployment explicitly.

22. **The RTX 5090 claim is unmeasured.** Report (l.237–238): "RTX 5090 and RTX PRO 4500 share the cause … consistent with GeForce's consumer tensor rates."
    - The 5090 profile differs (`moe.mxfp4_mma_from = 21`, `moe.tc5 = true`), and nothing was measured.
    - Mark this as a hypothesis.

23. **Line references that are close but not exact.**
    - `declare.rs:818` is the start of `Precision::ACTED`; the MXFP4 gran-32 entry is ~l.846.
    - `w4a8_mxfp4_mxfp8.py:36-41` binds `quantize_input = quantize_input_mxfp8`, not `mxfp4… = mxfp8_quantize`. The latter is the op it calls, at base.py:484–486.
    - There are two `fn select_act_quant` (dispatch/linear.rs:2026 and :2054, cfg variants); :2054 is the right one.
    - `scripts/…` paths mix the pie repo (`scripts/bench/vllm_bench.py`) and /root/r2g/scripts (`trtllm_pbatch.py`, `kern_window.py`). Use `results/scripts/` for the latter.

### OK, with a note

24. **Spare-change items, all minor:**
    - "+49 ms" for everything else: the unrounded figure is 49.6.
    - "pie's fa2 prefill is 16% faster": it uses 16% less time (1.19× faster).
    - vLLM 120b best is given as "29.2k" at l.229 but 29.3k in the table.
    - TRT clocks "2265–2317": the before-snapshots at large points reach 2325–2332.
    - main's 448-token 120b trace includes a `moe_select_mxfp4_mma<4,1,…>` variant (42 ms) that ee9098's lacks. "Same kernels" is approximately true: MoE total +30 ms (1%).
    - The pipe prototype's MoE total is 28 ms *slower* than staged and its wall is ~1–2% lower. "No faster" is fair.
    - Classification: TRT `globalExpertPrefixSumKernel` (0.9 / 0.5 ms) lands in "other", not MoE glue. TRT's qkv bias sits in `applyBiasRopeUpdateKVCacheV2` (norms), pie's in the nvjet epilogue (dense). Both negligible. All three TRT `GroupProblemShape` kernels are MoE (block-scaled e2m1×e4m3), correctly classified.

## Overall verdict

The measurements are solid and reproducible:
- The baseline and component tables regenerate exactly from the raw files.
- The kernel classification is sound.
- The core attribution holds: at equal tokens ~97% of kernel-time gap is the MoE expert GEMMs, with TRT-LLM on a W4A8 CUTLASS SM120 grouped GEMM against pie's W4A16 register-dequant selects.
- The TRT-LLM source citations check out.

The problems are in the summary prose and the projection section:
- Several summary numbers are stale round-1 values or wrong: the pf32 range, "7–9%", 29.1k, "spreads <1.5%".
- The pie power claim is contradicted by the in-run samples: pie is also at the 600 W cap.
- The effort totals do not add up.
- Most importantly, the "precision alone closes both cells / ~44k / ~74k" projection is undercut by the report's own measured W4A8 projection (37.9–41.9k for 120b) and by an inconsistent 20b rate (~67k at the stated 517 TFLOP/s).

Fix those, and scope the site-run and ea800efeb/batching conclusions to what n=1 runs at the tested points can show. With that, the report is ready.
