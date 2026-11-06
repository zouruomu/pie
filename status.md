# status: 20261106-1706-h100-sxm-x1-qwen3-6-27b-batching

## Update 19:20Z (site-style prompts, per note 1)

- **Note 1 acknowledged.** (1) All runs from here use the site's prompt construction (anton-mel/stats `synthetic.py` prompt_words_for_tokens(128) + common.py's " (Request #i)", chat-wrapped: 156-159 tokens on Qwen3.6, 194-197 on Qwen3.8), one prompts file for every engine (`results/scripts/site_prompts.py`). The 167-token runs below are kept as exploratory. (2) TRT-LLM settings recorded and grid-searched for its best working config (below). (3) vLLM: see below.
- **TRT-LLM best working config on this card:** max_batch_size 128, max_seq_len 2048, KV block reuse off; defaults otherwise (free_gpu_memory_fraction 0.9, capacity scheduler GUARANTEED_NO_EVICT, max_num_tokens 8192, CUDA graphs on, bf16 GDN state). Qwen3.6 c128 2867, c256 2893, c512 **2897** tok/s, ITL p50 30.1 ms. Grid (results/trt/grid): its defaults do not start (max_batch_size 2048 needs 160 GB of state slots; with max_seq_len at its 65,536 default the warmup OOMs at every batch cap tried); batch caps 160/192/224 give the same c128 (2860-2870) and collapse at c256+ (1550-2190, ITL ~70 ms: the slots leave too few kv pages); 256 admits a couple of requests at a time (89 tok/s).
- **pie main on the same prompts:** c64 1990, c128 2207, c256 **2211**, c512 2092 → **−23.7% best to best** (Qwen3.6, round 1). ~72 lanes a fire against TRT-LLM's 128.
- **vLLM 0.32 with the same max_model_len (2048):** one block pool shared by mamba states and attention pages, attention blocks taken on demand and preempted when short; max_model_len only has to fit one request (`vllm/v1/core/kv_cache_utils.py:1395-1430`, `config/compilation.py:1503-1525`). So neither vLLM nor TRT-LLM reserve the declared context per running sequence; pie's load fit does, for hybrids.
- **Site commit ee9098f60 vs main** (2 rounds, 167-token prompts): c128 2171-2177 vs 2175-2180; c256 2004-2015 vs 2191-2208. No regression on main.
- Running: slot-multiplier experiment engine (how many slots a sequence-at-declared-context the pool may seat; results/exp-seats), then TRT-LLM best config on Qwen3.8 + Qwen3.6 full curve, then Qwen3.8 pie/vLLM baselines.

## Early verdict (Qwen3.6-27B, 1 round; Qwen3.8-27B and more rounds pending) — 18:46Z

**Red on main, and the cause is pie's state-slot sizing, not decode speed.** Same prompts (167-token chat-rendered, 2c requests at concurrency c, 128 out, greedy, ignore_eos), SM 1920-1980 MHz:

| tok/s | c64 | c128 | c256 | c512 | c1024 |
|---|---|---|---|---|---|
| pie main dc6282d76 | 1962 | **2197** | 2191 | 2086 | 1802 |
| vLLM 0.32.0 | 1756 | 2139 | **2159** | fails to start (336 mamba blocks) | fails |
| TRT-LLM 1.4.0rc2.post1, max_batch_size 128, KV frac 0.70 | – | 2674 | 2816 | **2848** | – |
| pie main, `--max-model-len 512` (experiment) | – | **2751** | 2477 | 2474 | – |

- pie vs TRT-LLM best-to-best: 2197 vs 2848 = −22.9% (TRT-LLM grid with max_seq_len 2048 at batch caps 160-224 still running: its best may rise).
- pie ≈ vLLM (+1.7%). No regression since the site's commit: ee9098f60 2177/2004 vs main 2180/2191 at c128/c256 (round 1).
- All my absolute numbers (pie, vLLM) sit ~12% under the site's: the site's request count / prompt length differ from mine (its harness is private). Ratios are what I compare.
- **Lanes:** pie decodes ~72 lanes a fire (runtime `wave avg active pipelines`) at ITL p50 28.5 ms; TRT-LLM up to 128 at 30 ms. Step time per lane is alike: the gap is lanes.
- **Why pie seats ~72-92:** its load fit (`Pools::fit_the_card`, crates/engine-cuda/src/store.rs:1130-1163, `balanced_state_slots`, crates/engine/src/fit.rs:152) cuts the recurrent-state slots until its kv pages hold *every slot at the declared context* (2048 tokens, 128 MiB of kv a sequence), while these sequences use ~295 tokens. c128: 256 slots asked, 92 served, kv pages for 47×2048 = 96k tokens of which ~27k are ever used. TRT-LLM sizes slots = max_batch_size and gives the rest to pages (`mamba_cache_manager.py:4190-4200`), then admits by each request's own prompt+max_tokens (GUARANTEED_NO_EVICT, `llm_args.py:4319`). With a 512-token declared context pie serves 134 slots, ~108 lanes a fire, 2751 tok/s (+25%), above TRT-LLM's c128.
- **Fairness:** TRT-LLM and pie both keep the GDN state in bf16 (pie `models/qwen_3.poem:1105-1106`; TRT-LLM ignores the checkpoint's `mamba_ssm_dtype: float32`, `config_utils.py:300-325`); vLLM keeps fp32. KV reuse off for TRT-LLM; same prompt ids; 128 tokens each (ignore_eos). TRT-LLM at its defaults does not start on 80 GB (max_batch_size 2048 → 160 GB of state slots; warmup at its default max_seq_len 65,536 OOMs); I set max_seq_len 2048 (= pie's/vLLM's max_model_len) and grid its batch cap.
- Second, smaller effect: pie sizes the fire's working set for max_forward_requests = c lanes (pie_bench), so the cache room and slots shrink as c grows (c64 92 → c1024 66 slots) though only ~90 can seat.

## Now
- Running: site-commit A/B round 2, then TRT-LLM grid (max_seq_len 2048, batch caps 160/192/224/128).
- Next: Qwen3.8-27B sweep (pie, TRT-LLM best config, vLLM); fix prototype for the slot/page split; nsys step tables.
- Hand-over ETA ~01:15Z.
