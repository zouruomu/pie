# L4 × gpt-oss-20b (mxfp4): batching and pbatch vs vLLM 0.32.0

Run `20261103-2218-l4-x1-gpt-oss-20b-batching` on one NVIDIA L4 (24 GB, sm_89, 72 W power cap)
on RunPod. Fix branch: `r2g/20261103-2218-l4-x1-gpt-oss-20b-batching`. Evidence branch:
`r2g-evidence/…`. Every number below comes from a JSON in `results/`, and every table is regenerated
by a script in `scripts/` (§14).

## 1. Summary

* **pbatch is green on current main, and the site's pie value does not reproduce.** On this pod,
  pie's best pf-N is {{PF_PIE}} against vLLM's best {{PF_VLLM}}. The measured commit 8fb7e90e6 also
  gives pf8 7,325 here. The site's 4,666 (pf8, TTFT 1,045 ms) is not this engine's behaviour on an
  L4: the same commit's pf8 TTFT here is 661 ms. vLLM's site value (6,425) does reproduce (6,590).
* **Batching is level on current main.** Pie's best point is {{B_PIE}} and vLLM's {{B_VLLM}}
  (both at c512), a {{B_GAP}} gap with the spread shown in §3. Two things explain the site's −28%:
  * The site's vLLM batching row (2,193) does not reproduce here. With the harness defaults vLLM gives
    1,763 at c1024 and 1,790 at its best, with an ITL p50 of 112–121 ms against the site's 87.7.
    pie's site rows do agree with this pod: 8fb7e90e6's c512 ITL is 58 ms here against 63 on the site,
    and its raw TTFT 118 ms against 129. The site's vLLM row has TTFT and latency, so it came from vLLM's streaming client, as here
   (the non-streaming path records neither, `vllm_bench.py:318`); what else differed is not
   recoverable from the site.
  * pie improved after the measurement. #1350 (the L4's 65% working set, merged after the site
    measured 8fb7e90e6) alone takes the measured commit from c512 1,436 to 1,714 and from c256 1,504
    to 1,939 on this pod.
  #1368's admission cap then costs c256 13% (1,987 → 1,725), which is why main peaks at c512 and
  not c256. That cap is the remaining batching mechanism (§11). It is engine-wide, and no L4 profile
  setting recovers it without losing c512.
* **One profile line fixes a real red stretch of the batching curve.** At c8 and c16, pie trailed vLLM
  by 13% and 32%. Those decodes route 32 and 64 routes over gpt-oss-20b's 32 experts. The L4's
  `moe.mxfp4_mma_per_from = 4` sent them to the by-route MXFP4 select, which reads an expert's codes
  once for every route it serves: 40 ms of a c16 step whose ITL p50 is 50 ms. With `= 0` they take the register-dequant
  run kernel, which reads each touched expert once (22.5 ms):
  * c16: {{C16_BASE}} → {{C16_FIX}} (vLLM {{C16_VLLM}});
  * c8: {{C8_BASE}} → {{C8_FIX}} (vLLM {{C8_VLLM}});
  * the rest of the curve, raw and pbatch are unchanged.
  * Cross-model: gptoss-20b-mini c8 +12%, a 32-expert MiMo-V2.6-Flash miniature c8 +37% and c4 +7%,
    nothing slower.

## 2. How the site computes each metric

The GPU harness (pie-project/benchmarks) can't be read with this token: `gh repo clone` returns
"Could not resolve to a Repository". I reconstructed what it reports from three sources:

* **`scripts/bench/common.py:211-255` (`summarize`):** `output_tok_per_s = Σ output tokens / wall`,
  with TTFT and latency percentiles over requests.
* **anton-mel/stats, `src/pie_evals/node/engines/base.py:100-150` (`perf_from_common_json`):** this is
  the Mac harness, and it drives the same `common.py` client.
  * `prefill_tok_s` = median over requests of `prompt_tokens / ttft`;
  * `decode_tok_s` = median over requests of `(n-1)/(latency-ttft)`.
* **The site's own rows:** `data/db.js`, snapshot `l4-full`, db updated 2026-11-03T22:01Z, checked
  for internal consistency (`scripts/site_rows.py`):

| row | prefill_tok_s / output_tok_s | output_tok_s × wall (wall ≈ TTFT p99) | reading |
|---|---|---|---|
| pie pbatch | 4666.45 / 7.66 = **609.2** tokens a request | 7.66 × 1.057 s = **8.1** requests | **pf8** |
| vLLM pbatch | 6424.65 / 10.58 = **607.2** | 10.58 × 6.043 s = **63.9** | **pf64** |
| SGLang pbatch | 558.40 / 0.92 = **607.0** | 0.92 × 34.29 s = **31.5** | **pf32** |

**pbatch.** Three things follow for `prefill_batch`:

* The site's `prefill_tok_s` is `Σ prompt tokens / wall`. The per-request median of `prompt/ttft`
  would be 609 / 1.045 s = 583 tok/s, not 4,666.
* The prompt is about 609 chat-rendered tokens: 512 plus gpt-oss's harmony preamble.
* **Each engine's cell is its own best pf-N:** pf8 for pie, pf64 for vLLM, pf32 for SGLang.

**Batching (`throughput`).**

* **pie:** 1,576.34 tok/s at a latency p50 of 40.1 s means each request holds a seat for about 41.6 s.
  1,576 × 41.6 / 128 ≈ 512 requests in flight, so the cell is **c512** (1,024 requests, wall ≈ 83 s).
* **vLLM:** 2,192.91 tok/s with a latency p99 of 119.5 s. vLLM's client submits every request at t=0
  (`vllm_bench.py:538`), so its wall ≈ p99 latency, and 2,192.91 × 119.5 / 128 = **2,047 requests**:
  c1024 (2 × 1,024).
* Again, each engine's cell is its best point.

The two engines' latencies are not comparable:

* pie's client holds `--concurrency` requests open itself (`pie_bench.py:1100-1110`), so a request's
  clock starts when it gets a seat.
* vLLM's requests queue inside the engine (`max_num_seqs = concurrency`, `vllm_bench.py:373-376`), and
  its TTFT includes that queue.

The brief's "pie has better latency and ITL" comes from this difference, not from the engines.
pie's `raw` cell: 4,680.97 tok/s × 128.93 ms = 603.5 tokens, i.e. `prompt_tokens/ttft` per request.

**Setup here.**

* **Prompts:** built with `scripts/make_prompts.py` to exactly 225 chat-rendered tokens (batching) and
  609 (pbatch, raw), using the HF tokenizer.
* **Same tokens on both engines:** pie runs `--pretokenized-prompts`, so both engines prefill identical
  token ids. pie's own harmony rendering of the same text is 545 tokens (§12).
* **Shapes:**
  * batching: `tput --num-requests 2N --concurrency N --max-tokens 128`;
  * pbatch: `tput --num-requests N --concurrency N --max-tokens 1`;
  * raw: 8 requests one at a time, 256 out. vLLM's raw uses its streaming client at concurrency 1,
    since its latency mode stamps no TTFT.
* **Common flags** (harness defaults, both engines): `--warmup 2`, `--gpu-mem-util 0.90`,
  `--max-model-len 2048`.
* **Wrappers:** `scripts/bench.sh` and `scripts/sweep.sh`. Each run also samples SM clock and power
  once a second (`<label>.clk`).

## 3. Reproduced baseline on current main (48a36f0d8, before the rebase)

**Conditions.**

* Engine `base-48a36f0d8`, sha256 `07db7e20…` (recorded in every pie JSON).
* vLLM 0.32.0, which picks Marlin MXFP4 MoE, TRITON_ATTN, chunked prefill at 2,048 tokens and a KV
  pool of 92.7–98.3k tokens.
* One engine at a time on an idle GPU.

Rounds:

* **Round 1:** `results/base/`.
* **Rounds 2–3:** `results/verify2/` (r1 and r2 in the file names), alternating base, fix, vLLM.
* **The fix's round 1:** `results/verify/`.

{{TABLE_CURVE}}

SM clocks under load (median of 1 s samples above 40 W, `scripts/clocks.py`): {{CLOCKS}}. The L4
holds its 72 W cap under these loads, so its SM clock moves between about 900 and 2,000 MHz with load
and temperature. That spread is the run-to-run noise in the table.

## 4. What changed since the measured commit (8fb7e90e6)

All rows below were measured on this pod with engines built at each commit (sha256 in each JSON).
"+ws65" is 8fb7e90e6 with only `engine.working_set_percent = 65` added as a user profile layer, which
is exactly what #1350 (e1ef5b39a) changed.

| shape | 8fb7e90e6 | 8fb7e90e6 + ws65 (#1350 alone) | ee9098f60 (#1368's parent) | main 48a36f0d8 | main, ws 50% | main, seated_first off |
|---|---|---|---|---|---|---|
| pf8 | 7,325 | – | – | 7,693 (2) | 7,727 | – |
| pf256 | 4,904 | 7,676 | – | 7,553 | – | – |
| c16 | 297 | – | – | 298 (3) | – | – |
| c128 | 1,684 | – | – | 1,745 | – | – |
| c256 | 1,504 | 1,939 | 1,987 (2) | 1,725 (4) | 1,732 | 1,991 |
| c512 | 1,436 | 1,714 | 1,700 (2) | 1,785 (4) | 1,649 | 1,574 |
| c1024 | 1,409 | – | 1,705 (2) | 1,717 (3) | – | – |

(Mean of n runs where n > 1; `results/old`, `results/attr`, `results/knobs`, `results/knobs2`,
`results/base`, `results/diag`.)

**#1350, the L4's 65% working set** (merged 2026-11-02 19:24 PST, after the site measured 8fb7e90e6).

* At 50%, gpt-oss-20b's 8192-row fires were served at 1,024 rows and pf256 at 128 lanes
  (`old/pie-old-8fb7e90e6-pf256-r1.log`: "max_forward_tokens 1024 and max_forward_requests 256 are
  served at 1024 and 128").
* At 65%, pf256 keeps 4,871-row fires: **+57% pf256, +29% c256, +19% c512.**
* This is the change that closed most of the gap. The site has not measured pie on L4 since it landed.

**#1368, the admission cap** (`crates/runtime/src/bootstrap.rs:317-341`).

* A seated-first pool admits at most five sequences of the declared context: `5 × pages × 16 / 2048`.
* At c512 the pool has 6,528 pages, so the cap is 255 (`diag/pie-c512-base-info.log`: "admission:
  capped … requested=512 seated=255"), and c512 gains about 4.5% (1,700 → 1,785).
* At c256 the fire is 8,192 rows wide (lanes ≤ 256 leave the working set room), the pool shrinks to
  5,504 pages, and the cap is 215.
* The harness offers 512 requests that all have the same length, so they run in lockstep waves of
  215 + 215 + 82. The last wave decodes at a third of the width: on average 143 active lanes against
  200 before #1368 (`wave avg active pipelines` in the JSONs), and **c256 loses 13%**
  (1,987 → 1,725).
* Turning `seated_first` off removes the cap: c256 1,991, but c512 falls to 1,574 because 512 admitted
  lanes thrash the pool. Nothing in the L4 profile recovers c256 without losing c512 (§11).

## 5. Attribution: where pie's batching step time goes, and how it compares with vLLM's

**c512 (the batching cells' point).** nsys graph-level trace, 15.2 s of the measured window
(`results/prof/prof-c512-base-graph.graphs.txt`, `.gaps.txt`):

* The device is busy 15,016 of 15,208 ms (98.7%). Host-side gaps total 192 ms, one of which is 154 ms.
  pie is device-bound at c512, and host launch is not a factor.
* Decode steps (graph 47, about 254 lanes) take 72 ms: 96 runs, 6,912 ms.
* Prompt and mixed fires (graphs 116, 1, 20 and 35) take 42–190 ms each: 7,770 ms.
* So the window splits about 52% decode, 46% prefill or mixed, and 1.3% idle.

vLLM's ITL p50 at c512 is 121 ms, against pie's 84 ms. vLLM's steps carry chunked prefill (2,048
tokens) and about 220–262 running sequences, while pie's carry about 254. Both engines move the same
work through a pool of the same size:

* pie: 6,528 pages × 16 = 104k full-attention tokens, capped at 255 seats;
* vLLM: 98.3k tokens, about 262 sequences of 353.

They finish within a few percent of each other (c512: {{C512_GAP}}). There is no per-component
difference here big enough to attribute beyond the round-to-round spread.

**c16 (the fix).** nsys node trace, about 1.1 s of decode (`results/prof/prof-c16-*.kernels.txt`):

| per decode step (≈20 steps in the window) | base | fix | vLLM |
|---|---|---|---|
| MXFP4 selects (48 a step) | `moe_matmul_select_mxfp4` 0.84 ms × 48 = 40.3 ms | `moe_select_mxfp4_mma_run` 0.47 ms × 48 = 22.5 ms | Marlin `fused_marlin_moe` (`vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:224`) |
| step (ITL p50, `knobs/` 2 rounds) | 50.2 ms | 31.3 ms | 30.0 ms |
| the rest of the step (ITL − selects) | 9.9 ms | 8.8 ms | – |

The MoE delta (−17.8 ms a step) accounts for 94% of the step delta (−18.9 ms).

Why the by-route select costs that much:

* `mxfp4_mma_spans` (`crates/kernels-cuda/src/linear/moe/select_fp4.rs:27-32`) takes the
  register-dequant kernels only when `ceil(routes / experts) >= kernel.moe.mxfp4_mma_per_from`.
* The L4 file (`crates/engine-cuda/profiles/sm89/nvidia-l4.toml:38`, copied from the L40S) sets 4.
* Below that, `matmul_select_mxfp4` groups only past `mxfp4_grouped_reuse` = 2 routes an expert
  (`select_fp4.rs:907`). Otherwise it runs `moe_matmul_select_mxfp4`
  (`crates/kernels-cuda/kernels/linear/moe/select.cu:434`), which streams the routed expert's codes
  once per route.
* At c16, 64 routes touch about 28 of the 32 experts: 64 expert reads against 28, about 2.3× the bytes
  on a 300 GB/s card.
* vLLM selects Marlin for MXFP4 on this card (`Using 'MARLIN' Mxfp4 MoE backend`,
  `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:614`) at every batch size.

| lanes | routes | ceil(routes/32) | base | fix |
|---|---|---|---|---|
| 4 | 16 | 1 | under `mxfp4_mma_from` (28): by-route | same |
| 7–16 | 28–64 | 1–2 | by-route (≤ 2 × 32 routes) | run kernel |
| 17–24 | 68–96 | 3 | grouped wmma (`moe_matmul_select_mxfp4_wmma`, > 2 × 32 routes) | run kernel |
| 25–128 | 100–512 | 4–16 | run kernel | same |
| ≥129, and prefill (a 609-token prompt: 77 an expert) | > 512 | > 16 | staged kernel | same |

## 6. The fix, line by line

`crates/engine-cuda/profiles/sm89/nvidia-l4.toml`:

```diff
-# MXFP4 experts averaging 4 to 16 routes on the 512-code-run select, past
-# them the staged one (gpt-oss-120b's miniature at c128: 2490 to 2645 tok/s).
-moe.mxfp4_mma_per_from = 4
+{{COMMENT}}
+moe.mxfp4_mma_per_from = 0
```

* **What `0` does.** It means "any average". Since `per = ceil(routes/experts) >= 1`, it is the same
  as 1. With it, every bf16-activation MXFP4 select that already passes `mxfp4_mma_from` (28 routes,
  sm89 tier) takes the register-dequant kernels:
  * the 512-code run kernel up to `mxfp4_run_until` (16 a route average);
  * the staged kernel past that.
* **Which selects it changes.** Only selects of 28 or more routes that average fewer than 4 routes an
  expert:
  * gpt-oss-20b decodes of 7–24 lanes (28–96 routes), table in §5. 7–16 lanes come off the by-route
    select and 17–24 off the grouped wmma one; both move to the run kernel. c24 is measured in §7 for
    the second group.
  * a prefill or chunk of 7–24 tokens.
  Decodes of 25 lanes or more, and ordinary prefills (a 609-token prompt averages 77 routes an
  expert), are not touched.
* **Precedent.** The A40 profile has run `moe.mxfp4_mma_per_from = 0` since #1351. sm_86 has no run
  kernel, so there its selects land on the register-dequant `moe_select_mxfp4_mma` and staged kernels.
  The RTX 4090 still has `= 4` (`nvidia-geforce-rtx-4090.toml:41`), so its gpt-oss c8 and c16 decodes
  probably take the by-route select too (§12; not measured here).
* **A/B for this line:**
  * c16: 300 → 490 tok/s (knob A/B, `results/knobs/`, 2 rounds) and {{C16_BASE}} → {{C16_FIX}} in the
    sweep;
  * c8: 243 → 278 (knobs) and {{C8_BASE}} → {{C8_FIX}} (sweep).
  * `per0` and `per1` measure the same (`results/knobs/pie-per0-*`, `pie-per1-*`).
* **The rest of the comment** (the 512-code run select up to 16 and the staged one past it) is kept,
  because the run/staged split is unchanged.

## 7. Before / after (current main, 3 rounds each, alternating)

{{TABLE_BEFORE_AFTER}}

## 8. Token equality

Full greedy token-id lists were compared request by request (`scripts/tokeq.py` and
`scripts/agree.py`, both on `--dump-all-token-ids` JSONs).

* **raw, pf8 and every shape whose kernel choice does not change:** identical between base and fix, in
  every round, on all three models. The gpt-oss-20b shapes are listed in {{TOKEN_IDENTICAL}}.
* **Concurrency shapes:** pie's greedy outputs are not reproducible run to run even on one engine,
  because which lanes share a step changes the batch each kernel sees. Base against base across rounds
  is itself the noise floor:

{{TABLE_TOKENS}}

* **c8 and c16 on gpt-oss-20b:** the fix changes the kernel of every decode step, so outputs diverge,
  as two different kernels' accumulation orders would. Why this is acceptable:
  1. The run kernel is the one pie already decodes gpt-oss-20b with from c25 to c128, the widths every
     batching cell above c16 runs at.
  2. `the_routed_selects_answer_the_routed_dot`
     (`crates/kernels-cuda/tests/kernels_cuda/linear/moe_select.rs:199`) checks these kernels against
     the host's routed dot:
     * the by-route select at 1–2 routes an expert;
     * the register-dequant kernels at 1, 6, 8 and 12 (on sm_89 the run kernel, `[8, 1, 4]` and up);
     * the staged kernel at 50.
     It passes on both builds (§10). No case covers the run kernel at 2–3 routes an expert.
  3. Agreement with vLLM's greedy tokens on the same prompts is no worse with the fix than with base:
     {{VLLM_AGREE}}.

## 9. Cross-model, load and memory

**Same architecture, other sizes.**

* **gpt-oss-120b** cannot be checked on an L4. pie's only 120b miniature,
  `gptoss-120b-mini-e64` (`models/gpt_oss.poem:13`), keeps all 36 layers, about 31 GiB. A 4-layer cut
  is refused: "this model is built for 36 or more".
* **pie's gptoss-20b-mini** (5 layers, 16 experts, `shrink_checkpoint.py --layers 0-4 --experts 16`),
  2 rounds, `results/xmodel/oss20mini-*`:

| shape | base | fix |
|---|---|---|
| c4 | 494.5 | 495.9 |
| c8 | 793.6 | **888.1** (+12%) |
| c16 | 1,591.2 | 1,585.4 |
| c32 | 2,737.3 | 2,726.9 |
| c64 | 4,582.8 | 4,577.1 |
| pf8 | 36,695 | 37,140 |
| raw decode | 141.3 | 141.3 |

**Other MXFP4 architectures on the L4's catalog.**

* **MiMo-V2.6-Flash-RL**, layers 0–5 as the site cuts it. With all 256 experts it does not fit an L4
  at `device_memory_share` 0.9: the weights take 21.1 GB of the 21.3 GB allowed. The site's row says
  4.1B parameters, so its artifact keeps fewer experts, and how many is not on the site.
* So I measured a 32-expert cut, which exercises the line: c8 is 64 routes over 32 experts.
  Results, 2 rounds (`results/xmodel/mimoe32-*`):

| shape | base | fix |
|---|---|---|
| c4 | 309.1 | **330.6** (+7%) |
| c8 | 463.3 | **636.7** (+37%) |
| c16 | 1,155.0 | 1,160.0 |
| c64 | 2,510.6 | 2,512.3 |
| pf8 | 15,696 | 15,336 (one run at 14,997; prefill selects are untouched by the line, see {{MIMO_PF8}}) |
| raw decode | 89.7 | 89.7 |

* MiMo's tokens: identical between base and fix at every shape and round, except c16 in round 1
  (30/32 identical, the same as base against base).
* **deepseek-v4-flash's 5-layer cut** was not run. Its MXFP4 experts reach the same selects, but the
  checkpoint's artifact (mlx bf16 base, mxfp4 scheme) and layer cut on the site are not documented.

**Load.**

* Full curves c4 to c1024 and pf4 to pf1024, 3 rounds: §7.
* raw TTFT and decode: §7.

**Memory fit.**

* The line changes no allocation. The run kernel's route order, offsets and work list come from
  `ctx.scratch` (`crates/kernels-cuda/src/linear/moe/route.rs:487`), the same scratch the grouped, run
  and staged selects already take at c17 and up and in every prefill. It needs no copy of the bank.
* Load logs on both engines report the same pool, bodies and caps (`device_pages`, `seated`,
  "bodies armed … MiB") at every shape. {{MEMFIT}}

## 10. Tests

Both builds (base 48a36f0d8 and the fix, then the rebased pair) ran the same binaries' suites
single-threaded (`--test-threads=1`) on this L4:

{{TESTS}}

## 11. Remaining gap, broken down

With the fix, against vLLM 0.32.0 on this card, from §7's table:

| point | fix vs vLLM | mechanism | where |
|---|---|---|---|
| batching cell (each engine's best: c512 for both) | {{GAP_B}} | **c256 admission.** #1368's cap seats 215 of 256, so 512 lockstep requests run in waves of 215 + 215 + 82 (§4). Without the cap c256 reaches 1,987–1,994 (#1368's parent, and main with `seated_first = false`), about 11% over vLLM's best. The cap's factor of five is engine-wide (measured on A100 Gemma-4), and on this card c512 needs it: without it, c512 is 1,574. | `crates/runtime/src/bootstrap.rs:318-341` |
| c512, c1024 | {{GAP_C512}}, {{GAP_C1024}} | Equal pools (pie 255 seats, vLLM about 262) and device-bound steps (98.7% busy, §5). Nothing to attribute beyond the spread. | – |
| pf512, pf1024 | {{GAP_PF512}}, {{GAP_PF1024}} | `max_forward_requests = 512` reserves a working set for 512 lanes, so fires are served at 1,024 rows instead of 8,192 (§12.5). Not part of the pbatch cell, where each engine's best is pf8–pf128. | `crates/engine-cuda/src/serve/load.rs:324, 589` |
| c4, c8, c16 | {{GAP_C4}}, {{GAP_C8}}, {{GAP_C16}} | Level or ahead after the fix. | – |

The pbatch cell has no remaining gap: pie leads at every pf-N up to pf256.

## 12. Side findings

1. **Two of the site's L4 cells don't reproduce on this pod.**
   * pie's pbatch at 8fb7e90e6: pf8 4,666 there (TTFT 1,045 ms), 7,325 here (TTFT 661 ms) for the same
     commit.
   * vLLM's batching: 2,193 there (ITL 87.7 ms), 1,763 at c1024 and 1,790 at best here (ITL 112–121 ms).
   The other rows agree within about 10%: pie raw, pie batching at 8fb7e90e6 (1,576 there, 1,436 here
   at c512), vLLM raw and vLLM pbatch. The site's vLLM row has TTFT and latency, so it came from vLLM's streaming client, as here
   (the non-streaming path records neither, `vllm_bench.py:318`); what else differed is not
   recoverable from the site.
2. **The site's batching and pbatch cells pick each engine's own best concurrency.** pie is pf8
   against vLLM's pf64, and c512 against c1024. This should be stated on the site.
3. **TTFT and latency are not comparable across engines under concurrency.** pie's client holds the
   queue, while vLLM's engine does. vLLM's TTFT p50 at c4 is 1.75 s because of its queue, not its
   prefill.
4. **pie's harmony rendering of the bench prompt is 545 tokens, HF's is 609.** pie's template omits
   the "You are ChatGPT…/Knowledge cutoff/Current date/Reasoning" system preamble. Without
   `--pretokenized-prompts`, pie prefills 11% fewer tokens than vLLM on the same prompt strings.
5. **pf512 and pf1024 trail vLLM by 24% and 19%** (4,994 and 5,259 against 6,543 and 6,510).
   * `max_forward_requests = concurrency` (`pie_bench.py:297-300`) asks for 512 lanes.
   * The working set at 512 lanes leaves room for only 1,024-row fires (`serve/load.rs:324`, "served
     at 1117" then "1024 and 512").
   * A prompt-only fire never carries more than ⌊8192/609⌋ = 13 lanes.
   * `engine.prefill_lanes` exists for this and is not set on the L4. It is untested here.
6. **The L4's `forward_tokens = 2304` does nothing for gpt-oss-20b.** Sparse-expert traces take
   `max(forward_tokens, 8192)` (`serve/load.rs:72-79`).
7. **Graph bodies at 512 MiB break arming for gpt-oss-20b** (1 of 1 compositions armed, every shape
   walking eagerly), the same as upstream found on the RTX 4090 (#1417).
8. **The RTX 4090 profile probably has the same c8/c16 gap.** It keeps `moe.mxfp4_mma_per_from = 4`
   (`nvidia-geforce-rtx-4090.toml:41`) on the same sm_89 kernels. Its #1355 raised `mxfp4_run_until`
   to 32, which only moves selects of 17–32 routes an expert. Not measured here.
9. **nsys exports carry the process environment, including API keys.** The traces stay on the pod
   (§13).
10. **vLLM pf16 round 1 read 4,290 against 6,150–6,590 elsewhere on its curve.** {{VLLM_PF16}}

## 13. Playbook notes

* **The site picks each engine's best point of the sweep.** Reconstruct which point from
  `output_tok_s × wall` before profiling anything. Here the pbatch cell was pie pf8 against vLLM pf64.
* **Build the measured commit and run it on the pod first.** It separates the site's environment from
  engine changes. Here, the same commit gave 7.3k against the site's 4.7k.
* **Under N = 2C lockstep batching, small differences in seat count become wave quantization.** 215
  seats for 512 requests means three waves. Read `wave avg active pipelines` and the info-level
  admission log (`RUST_LOG=info`).
* **Never commit nsys `.nsys-rep` or `.sqlite`.** They contain the process environment, keys included.
  GitHub push protection caught a RunPod key in one here. Keep traces outside the evidence tree and
  commit text summaries.
* **`--cuda-graph-trace=graph` exports carry graph executions from before `cudaProfilerStart`**
  (negative timestamps, load-time arming). Filter `start >= 0`.
* **No CPU sampling on these containers:** py-spy has no ptrace, and perf_event_paranoid is 4.
* **The L4 is power-capped.** SM clocks swing between 900 and 2,000 MHz with load and temperature, so
  sample clocks during each run, not after it.

## 14. Files

TBD
