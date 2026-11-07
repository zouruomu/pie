# Brief: 20261107-0105-h100-sxm-x1-qwen3-6-27b-batching

**This is a retry of `20261106-1706-h100-sxm-x1-qwen3-6-27b-batching`**, which reproduced the gap and found its cause in 3 h, then died when its Claude token was revoked (not its fault). Its brief (`runs/20261106-1706-h100-sxm-x1-qwen3-6-27b-batching/brief.md` on r2g-control) holds the full cell tables, workload definitions, key code pointers and TRT-LLM setup notes: read it first; everything there still applies except the time plan. Its evidence is on branch `r2g-evidence/20261106-1706-h100-sxm-x1-qwen3-6-27b-batching` of zouruomu/pie: `git fetch origin r2g-evidence/20261106-1706-h100-sxm-x1-qwen3-6-27b-batching` and reuse its scripts (`results/scripts/`: site_prompts.py builds the site-style prompts, trtllm_tput.py drives TRT-LLM, bench.sh, sweep.sh, prof-*.sh) and its raw results as the baseline reference. Its status.md and `runs/20261106-1706-h100-sxm-x1-qwen3-6-27b-batching/outcome.md` summarise what it learned. Don't redo what it did unless you need the number on your own build.

## The cells (H100 SXM 80 GB HBM3, sm_90; site `h100-sxm-x1`)

- **Qwen3.8-27B × batching:** site pie 2311.30 (`225b6b94d`, 2026-11-03 06:03, stale tag h100-full) vs TRT-LLM 1.4.0rc2.post1 3356.51 (2026-11-05 21:01): −31.1%. vLLM 0.32.0 2350.63, SGLang 0.5.22 1704.49.
- **Qwen3.6-27B × batching:** site pie 2489.15 (`ee9098f60`, 2026-11-03 14:54, stale tag h100-fill) vs TRT-LLM 3366.33 (2026-11-05 21:55): −26.1%. vLLM 2417.75, SGLang 1644.56.
- Scheme bf16, full checkpoints `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.8-27B` (not miniatures). The site still scores these numbers (db of 2026-11-07 00:56Z).
- Workload (site `throughput`): 128-token prompts, 128 output tokens; sweep c4 … c1024; best point scored; mid-curve points ≥4.5% behind the best competitor at the same point are in scope.

## What the previous run established (on main dc6282d7; 1 round each; site-style prompts, ~158 / ~196 tokens chat-rendered)

| tok/s | c64 | c128 | c256 | c512 | c1024 |
|---|---|---|---|---|---|
| Qwen3.6 pie | 1990 | 2207 | **2211** | 2092 | 1816 |
| Qwen3.6 vLLM 0.32.0 | 1791 | 2151 | 2182 | fails to start | – |
| Qwen3.6 TRT-LLM (max_batch_size 128, max_seq_len 2048, reuse off) | 1978 | 2852 | 2914 | 2909 | **2916** |
| Qwen3.8 pie | 1890 | 2090 | **2097** | 1994 | 1734 |
| Qwen3.8 vLLM | 1699 | 2045 | 2063 | – | – |
| Qwen3.8 TRT-LLM | 1887 | 2651 | 2691 | 2703 | **2703** |

- Best to best: Qwen3.6 −24.2%, Qwen3.8 −22.4% vs TRT-LLM. pie ≈ vLLM. No regression since the site commits. TRT-LLM's defaults don't start on 80 GB; the config above is its best working one from a grid (batch caps 160-224 collapse at c256+).
- **Cause (its attribution):** for hybrid GDN trunks, pie's load fit (`Pools::fit_the_card`, crates/engine-cuda/src/store.rs:1130-1163; `balanced_state_slots`, crates/engine/src/fit.rs:152) cuts recurrent-state slots until the kv pages hold *every slot at the declared context* (2048 tokens, ~128 MiB of kv a sequence), while these sequences use ~290 tokens. pie decodes ~72 lanes a fire (c128: 256 slots asked, 92 served) vs TRT-LLM's 128, at alike ITL (28.5 vs 30.1 ms). TRT-LLM sizes slots = max_batch_size and gives the rest to pages, admitting by each request's own prompt+max_tokens (GUARANTEED_NO_EVICT); vLLM 0.32 shares one block pool between mamba states and attention pages, taking attention blocks on demand. Neither reserves the declared context per running sequence.
- **Experiments:** `--max-model-len 512`: c128 2751 (+25%), c256 2477, c512 2474. Seating k× the fitted slots (an unsaved experiment engine): c128 x1 2237, x3 2260, x5 2782, x8 1911; c512 x2 2305, x3 2444, x5 2376, x8 1833. Over-seating collapses, so a fix must keep pages and slots in balance by actual use, not by a multiplier.
- Second, smaller effect: pie sizes the fire's working set for max_forward_requests = c lanes, so cache room and slots shrink as c grows (c64 92 → c1024 66 slots).

## Your job

Turn that cause into a shippable fix and verify it. The open design question (orchestrator note 1 of the previous run): is reserving the declared context per slot a bug or a policy? Your fix should size state slots and kv pages by what sequences actually use (admit on each request's own prompt + max_tokens, or share the pool on demand with a correct preemption/pause path), not by a per-profile multiplier and not by lowering max_model_len. Keep it minimal (KISS); the fix most likely lives in fit.rs / store.rs / runtime admission (`crates/runtime/src/bootstrap.rs` `rs_seat_cap`), not in the H100 profile. If you conclude the right fix is a profile/policy change, say why with evidence.

Measure what matters for an admission/seating change (the A100 Gemma-4 run's lesson, ledger 2026-11-06): ITL p99 and pause/suspend events over 5+ runs at c512/c1024, not only throughput; long-context requests still fitting (a request that does use 2048 tokens must not fail or deadlock); non-hybrid models unchanged (one dense, e.g. Qwen3-8B or Gemma-4-31B, and one MoE); and whether it helps the related cells by the same mechanism.

## Related red cells (same models, batching; not on this pod — say whether the cause applies)
A100 SXM Qwen3.8 −27.9% / Qwen3.6 −27.2%; L40S (minis) −25.0% / −23.5%; A40 (minis) −21.3% / −17.1%; RTX 5090 (mini) Qwen3.8 −16.5%; H200 Qwen3.8 −10.3%; B200 Qwen3.6 −8.1% (vLLM); RTX 4090 (mini) Qwen3.8 −6.1%; PRO 4500 (mini) Qwen3.8 −76.8% (ledger: already fixed in #1362, ignore). The TPU cells are another engine.

## Freshness (checked 2026-11-07 01:05Z, main at 41f10ec)
Nothing on main since the previous run's base touches the H100 profile, fit.rs, store.rs, serve/load.rs or runtime bootstrap (only #1484, engine-xla GDN loop). Open PRs on pie-project/pie: none on Qwen, GDN, H100, state slots or admission. sync.sh lists new upstream commits at every check-in: if any touches state slots, fit, admission or the RS store, rebase and re-measure at once.

## Ledger notes
- engine-cuda GPU suite: build with CARGO_PROFILE_DEV_OPT_LEVEL=2, single-threaded, nothing else on the GPU meanwhile; start it as soon as the diff settles.
- Run `/root/r2g/sync.sh` yourself in the foreground at each check-in (notes printed by background scripts are lost).
- TRT-LLM 1.4 setup notes: PRO 6000 run's report.md "Playbook notes" (`r2g-evidence/20261106-0910-pro6000-x1-gpt-oss-120b-pbatch`) and the previous run's scripts.

## Time plan
- **Deadline:** 16 h from pod start (the pod terminates itself).
- **Early verdict, about 2 h in:** re-confirm the gap on current main at the best points (pie vs TRT-LLM, ≥3 rounds at c256 and c1024 for Qwen3.6, at least 1 for Qwen3.8; clocks), plus your fix design in status.md. The baseline is already known, so spend most of the first 2 h on the fix design and a prototype.
- **Fix prototype measured on both models over the sweep:** by about 5 h.
- **Target hand-over:** about 8 h in. Leave the rest for review rounds.
- Check in with sync.sh at least hourly; notes arrive as `runs/20261107-0105-h100-sxm-x1-qwen3-6-27b-batching/note-N.md`.

## Expected at hand-over (per the PLAYBOOK)
- `report.md`: baseline (commit, clocks, rounds), TRT-LLM settings, the lanes/slots/pages accounting before and after, root cause with pie `file:line` and TRT-LLM's/vLLM's `file:line`, the fix line by line, before/after over the c sweep for both models (≥3 rounds at the best points), token equality (base-vs-fix next to base-vs-base under concurrency), tail (ITL p99, pauses) over 5+ runs at c512/c1024, long-context fit, cross-model checks (dense + MoE), other loads (pbatch, raw, TTFT), memory fit, tests (logs in results/tests/, GPU suite on the fix build, failures rerun on base), remaining gap, side findings, playbook notes.
- `pr.md`: title in pie's style, one or two dense paragraphs, no attribution.
- `results/`: raw JSON and logs behind every number, engine sha256 per run, scripts.
