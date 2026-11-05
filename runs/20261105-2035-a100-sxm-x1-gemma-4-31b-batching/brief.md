# Brief: 20261105-2035-a100-sxm-x1-gemma-4-31b-batching

## The cell

A100 SXM 80 GB × Gemma-4-31B (bf16, full model, not a miniature) × batching. The site scores pie at −18.1% against vLLM.

| field | value |
|---|---|
| hardware | `a100-sxm-x1`, NVIDIA A100 SXM 80GB, 1 GPU (this pod: NVIDIA A100-SXM4-80GB) |
| model | `gemma-4-31b` (google/gemma-4-31B-it), dense, bf16, `miniature: false` |
| mode | batching (workload `throughput`), metric `output_tok_s`, each engine's best point of its sweep |
| pie | 465.76 tok/s, commit `d9c9582b6`, measured 2026-11-03 05:05 (site tag `a100-sxm-full`, `pie_stale` set) |
| best | vLLM 0.32.0, 569.00 tok/s, measured 2026-11-03 03:24 |
| gain | −18.1% |
| also measured | SGLang 0.5.22: 419.06 tok/s |

Detail from the site's rows at each engine's best point:

| engine | out tok/s | decode tok/s | TTFT p50 | ITL p50 |
|---|---|---|---|---|
| pie | 465.76 | 18.8 | 127.0 s | 46.58 ms |
| vLLM 0.32.0 | 569.00 | 14.93 | 215.7 s | 51.63 ms |
| SGLang 0.5.22 | 419.06 | 20.62 | 269.4 s | 47.12 ms |

Pie's ITL is lower than vLLM's, yet its throughput is lower: out tok/s × ITL gives about 22 sequences decoding at once for pie against about 29 for vLLM. So the first suspect is how many sequences pie seats, not step speed: KV pool size, the sliding-window layers' pages, admission, graph-body memory. Check that suspicion; don't assume it. With bf16 weights of ~58 GiB on 80 GB, the KV pool is small, and every GiB matters.

**About the measured commit:** `d9c9582b6` is the head of #1368's branch ("A100 profile: seated first"), not a main commit. #1368 landed on main as `c57e793`. Its own text gives Gemma-4-31B c128 309 → 465 tok/s, so the site's number already includes #1368's seated-first and its five-sequences admission cap (`crates/runtime/src/bootstrap.rs`, the `seated_first` cap). The gap that remains is after that fix.

## Workload

From the site's `workloads` table, `throughput`: concurrency 32 (headline), 128 prompt tokens, 128 output tokens; the sweep is c4, c8, c16, c32, c64, c128, c256, c512, c1024, and the site scores each engine's best point. The TTFTs above suggest the best points are high concurrencies (c512 or c1024); find out which. Read `scripts/bench/` (pie_bench.py, vllm_bench.py, sglang_bench.py, common.py) and `.github/workflows/evals.yml` for the exact flags, including `--max-model-len`/declared context, which the admission cap depends on.

## Related cells (check alongside if cheap; the target is the cell above)

- Gemma-4-31B batching is red on several cards: H100 SXM −24.4% vs vLLM (measured on `b2ec7e880`, before #1368), A40 mini −31.7% vs SGLang, RTX PRO 6000 −10.3% vs vLLM, RTX PRO 4500 mini −7.6%. H200 and B200 are green (more memory). If the cause is memory/seating, it probably reaches all 80 GB cards; say what in your fix would carry over, but measure only the A100.
- On this A100 the Gemma-4-31B pbatch (pie 3713.68 vs vLLM 3793.43, −2.1%), prefill and decode are green. Check that your fix keeps them so.
- Other red cells on this card: gpt-oss-120b batching −18.0% vs vLLM (measured on `ee9098f60`, before #1368, so possibly already moved); Nemotron-3-Ultra mini batching −8.6% (old, `e9251eb21`). Use them as cross-model checks if your fix touches seating or memory.

## Prior art on main (read these first)

- #1358 `e3e90d0` (RTX 5090 profile): `seat_to_end = true`, `window_spare = false` (no spare ring page a lane: "the spare pages took ~4 GB beside Gemma-4-31B mini18"), `recording = { use = "bodies", mem_mb = 1024 }` (graph bodies held to 1 GiB, not a quarter of what is past the weights), `state_slots_per_lane = 1`. Gemma-4-31B mini18 c128 2.39k → 2.94k.
- #1411 `1ec05c7` (RTX 4090 profile): `seated_first`, `seat_to_end`, `recording mem_mb = 512`. Gemma-4-31B mini12 c256 2.01k → 2.82k.
- #1368 `c57e793`: the A100 profile has `seated_first = true` only: no `seat_to_end`, `window_spare` at its default, `recording` at its default. The H100 profile has `seated_first` + `seat_to_end` but not the other two.
- Knob experiments need no rebuild: a user layer at `~/.pie/profiles/cuda/user/<device>.toml` (see the PLAYBOOK), deleted after each run.

## What has landed on main since the measured commit

`d9c9582b6` is 3 commits ahead of and 79 behind main (at `e4bb499`, 2026-11-05 19:34 UTC). Since #1368, nothing has touched `crates/engine-cuda/profiles/sm80*`. Commits that could change memory or seating for this cell:
- `1716873` #1400 Arena: the attention score planes held out of the placement, then lifted over the rest (`crates/poem-compiler/src/arena.rs`)
- `e4bb499` #1436 runtime: a piece writes only the working sets with rows in it (`crates/runtime/src/pipeline/fire.rs`)
- `0ea459a` #1418 prompt keys hashed outside the KV store lock
- `622bf42` #1315 cargo dependency bumps

So reproduce on current main first: the gap may have moved.

## Overlap to watch

pie's maintainers (anton-mel, ingim) land "… part of the pie vs X gap on <GPU>" PRs almost daily, including Gemma-4 admission on other cards. Another r2g run, `20261105-2033-h100-sxm-x1-gpt-oss-120b-batching`, is working on gpt-oss-120b on an H100 at the same time. It's a different model and card, but if your fix changes shared runtime code (admission, bootstrap), say so in status.md. sync.sh lists what lands on upstream main; rebase and re-measure if anything touches seating, the KV pools, or the A100 profile.

## Ledger notes from earlier runs

- The site metric is each engine's best point of its sweep; a red point in the middle of the curve is in scope too.
- #1368's admission cap cost the L4 13% at c256 (gpt-oss-20b), seen in run `20261103-2218`. If you touch the cap, check it on more than one model.
- engine-cuda GPU suite: build it as a dev build at `CARGO_PROFILE_DEV_OPT_LEVEL=2` (74 min on an L4; an A100 should be faster), run it with `--test-threads=1`, and run nothing else on the GPU while it runs.
- `the_routed_selects_answer_the_routed_dot` fails off sm_120a/121a at #1274's first W4A8 case; expect it on the base build too and don't attribute it to your change. Confirm on base anyway.
- Under concurrency pie's greedy outputs differ between reruns of one engine; compare base vs fix next to base vs base.

## Time plan

- **Deadline:** 16 h from pod start; the pod terminates itself then.
- **Early verdict:** about 2 h in. Reproduce pie (current main) and vLLM 0.32.0 over the whole sweep, at least c64–c1024 with 2–3 rounds, and put the verdict at the top of status.md: is the gap there, at which points, how large, how much spread. Include how many sequences each engine actually has decoding (pie's seated lanes and KV pool pages/GiB from its log; vLLM's KV cache size and max concurrency from its log).
- **Target hand-over:** about 8 h in. The rest is for review rounds.
- If the cell is already green on main and nothing else on the curve is red at 4.5% or more, write it up briefly and hand over early.

## What I expect at hand-over (as in the PLAYBOOK)

- `report.md` with: the reproduced baseline (commit, clocks, conditions); how many sequences each engine seats and why (pie's memory budget broken down: weights, KV pages, window/ring pages, graph bodies, working sets, anything else, against vLLM's); the cause with `file:line`; the fix line by line, with an A/B for every line; before/after tables over the whole sweep; token-id equality (full id lists; base vs fix next to base vs base); cross-model checks on this card (at least Gemma-4-26B-A4B or another Gemma-4, gpt-oss-120b batching, and one non-Gemma model); other loads (pbatch, prefill-heavy, TTFT); memory fit (does any model stop fitting); tests; remaining gap; side findings.
- `pr.md`: title and body in pie's style, no attribution.
- `results/`: every raw JSON and log behind every number, the engine sha256 in each run's JSON, and the analysis scripts.
